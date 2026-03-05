//! Pure Rust reader for NAIF DAF/SPK binary ephemeris files.
//!
//! Implements the subset of the DAF (Double Precision Array File) and
//! SPK (Spacecraft and Planet Kernel) specifications needed to read
//! Type 2 segments (Chebyshev position-only polynomials) from files
//! like JPL DE440.
//!
//! Reference: NAIF DAF Required Reading (daf.req)
//!            NAIF SPK Required Reading (spk.req)
//!
//! All positions are in km, times in seconds past J2000 TDB.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

// ── Constants ──────────────────────────────────────────────────────

/// DAF record size in bytes (128 doubles * 8 bytes).
const RECORD_BYTES: u64 = 1024;

/// Number of doubles per DAF record.
const RECORD_DOUBLES: usize = 128;

/// SPK files have ND=2, NI=6.
const SPK_ND: usize = 2;
const SPK_NI: usize = 6;

/// Summary size in doubles for SPK: ND + ceil((NI+1)/2) = 2 + 3 = 5.
const SPK_SS: usize = SPK_ND + (SPK_NI + 1) / 2; // 5

/// Maximum summaries per summary record: (128 - 3) / SS = 25.
#[allow(dead_code)]
const MAX_SUMMARIES_PER_RECORD: usize = (RECORD_DOUBLES - 3) / SPK_SS;

/// NAIF body codes.
pub const SOLAR_SYSTEM_BARYCENTER: i32 = 0;
pub const EARTH_MOON_BARYCENTER: i32 = 3;
pub const SUN: i32 = 10;
pub const EARTH: i32 = 399;
pub const MOON: i32 = 301;

// ── Error types ────────────────────────────────────────────────────

/// Errors that can occur when reading an SPK file.
#[derive(Debug)]
pub enum SpkError {
    /// I/O error reading the file.
    Io(io::Error),
    /// File is not a valid DAF/SPK file.
    InvalidFormat(String),
    /// Requested segment not found.
    SegmentNotFound { target: i32, center: i32 },
    /// Epoch is outside the coverage of the segment.
    EpochOutOfRange { epoch: f64, start: f64, end: f64 },
    /// Unsupported SPK data type (we only support Type 2).
    UnsupportedType(i32),
}

impl From<io::Error> for SpkError {
    fn from(e: io::Error) -> Self {
        SpkError::Io(e)
    }
}

impl core::fmt::Display for SpkError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SpkError::Io(e) => write!(f, "I/O error: {}", e),
            SpkError::InvalidFormat(msg) => write!(f, "invalid SPK format: {}", msg),
            SpkError::SegmentNotFound { target, center } => {
                write!(
                    f,
                    "no segment found for target={} center={}",
                    target, center
                )
            }
            SpkError::EpochOutOfRange { epoch, start, end } => {
                write!(
                    f,
                    "epoch {} outside segment range [{}, {}]",
                    epoch, start, end
                )
            }
            SpkError::UnsupportedType(t) => write!(f, "unsupported SPK data type {}", t),
        }
    }
}

// ── DAF file record ────────────────────────────────────────────────

/// Parsed DAF file record (first 1024 bytes).
#[derive(Debug)]
#[allow(dead_code)]
struct DafFileRecord {
    /// "LTL-IEEE" or "BIG-IEEE"
    endianness: Endianness,
    /// Number of double components in each summary (should be 2 for SPK).
    nd: usize,
    /// Number of integer components in each summary (should be 6 for SPK).
    ni: usize,
    /// Record number of the first summary record.
    fward: usize,
    /// Record number of the last summary record.
    bward: usize,
    /// First free DAF address.
    _free: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endianness {
    Little,
    Big,
}

fn read_file_record(file: &mut File) -> Result<DafFileRecord, SpkError> {
    file.seek(SeekFrom::Start(0))?;

    let mut buf = [0u8; RECORD_BYTES as usize];
    file.read_exact(&mut buf)?;

    // Bytes 0..8: identification word "DAF/SPK "
    let id_word = &buf[0..8];
    if &id_word[0..4] != b"DAF/" {
        return Err(SpkError::InvalidFormat(format!(
            "not a DAF file (got {:?})",
            core::str::from_utf8(&id_word[0..7]).unwrap_or("???")
        )));
    }

    // Bytes 88..96: endianness string
    let fmt_str = core::str::from_utf8(&buf[88..96])
        .map_err(|_| SpkError::InvalidFormat("bad format string".into()))?
        .trim();

    let endianness = if fmt_str.starts_with("LTL") {
        Endianness::Little
    } else if fmt_str.starts_with("BIG") {
        Endianness::Big
    } else {
        return Err(SpkError::InvalidFormat(format!(
            "unknown endianness: {}",
            fmt_str
        )));
    };

    let read_i32 = |offset: usize| -> i32 {
        let bytes: [u8; 4] = buf[offset..offset + 4].try_into().unwrap();
        match endianness {
            Endianness::Little => i32::from_le_bytes(bytes),
            Endianness::Big => i32::from_be_bytes(bytes),
        }
    };

    let nd = read_i32(8) as usize;
    let ni = read_i32(12) as usize;
    let fward = read_i32(76) as usize;
    let bward = read_i32(80) as usize;
    let free = read_i32(84) as usize;

    if nd != SPK_ND || ni != SPK_NI {
        return Err(SpkError::InvalidFormat(format!(
            "expected ND={} NI={}, got ND={} NI={}",
            SPK_ND, SPK_NI, nd, ni
        )));
    }

    Ok(DafFileRecord {
        endianness,
        nd,
        ni,
        fward,
        bward,
        _free: free,
    })
}

// ── Segment descriptor ─────────────────────────────────────────────

/// Parsed SPK segment descriptor.
#[derive(Debug, Clone)]
pub struct SegmentDescriptor {
    /// Start epoch (seconds past J2000 TDB).
    pub start_epoch: f64,
    /// End epoch (seconds past J2000 TDB).
    pub end_epoch: f64,
    /// NAIF target body code.
    pub target: i32,
    /// NAIF center body code.
    pub center: i32,
    /// Reference frame code (1 = J2000).
    pub frame: i32,
    /// SPK data type (2 = Chebyshev position only).
    pub data_type: i32,
    /// Initial DAF array address (1-indexed).
    pub start_addr: usize,
    /// Final DAF array address (1-indexed).
    pub end_addr: usize,
}

// ── SPK file handle ────────────────────────────────────────────────

/// An opened SPK file, ready for queries.
pub struct SpkFile {
    file: File,
    header: DafFileRecord,
    /// All segment descriptors, in file order.
    segments: Vec<SegmentDescriptor>,
}

impl SpkFile {
    /// Open and parse an SPK file from disk.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, SpkError> {
        let mut file = File::open(path)?;
        let header = read_file_record(&mut file)?;
        let segments = read_all_segments(&mut file, &header)?;

        Ok(SpkFile {
            file,
            header,
            segments,
        })
    }

    /// List all segment descriptors in this file.
    pub fn segments(&self) -> &[SegmentDescriptor] {
        &self.segments
    }

    /// Find the segment for a given target/center pair that covers the epoch.
    /// Returns the last matching segment (highest precedence per DAF rules).
    pub fn find_segment(
        &self,
        target: i32,
        center: i32,
        epoch: f64,
    ) -> Result<&SegmentDescriptor, SpkError> {
        // DAF precedence: last segment in file wins
        self.segments
            .iter()
            .rev()
            .find(|s| {
                s.target == target
                    && s.center == center
                    && epoch >= s.start_epoch
                    && epoch <= s.end_epoch
            })
            .ok_or(SpkError::SegmentNotFound { target, center })
    }

    /// Evaluate position (km) for a target relative to center at epoch (seconds past J2000 TDB).
    /// Returns [x, y, z] in the segment's reference frame (J2000/ICRF for DE440).
    pub fn position(&mut self, target: i32, center: i32, epoch: f64) -> Result<[f64; 3], SpkError> {
        // Find the segment
        let seg = self
            .segments
            .iter()
            .rev()
            .find(|s| {
                s.target == target
                    && s.center == center
                    && epoch >= s.start_epoch
                    && epoch <= s.end_epoch
            })
            .ok_or(SpkError::SegmentNotFound { target, center })?
            .clone();

        if seg.data_type != 2 {
            return Err(SpkError::UnsupportedType(seg.data_type));
        }

        evaluate_type2(&mut self.file, &self.header, &seg, epoch)
    }

    /// Evaluate position and velocity (km, km/s) for a target relative to center.
    /// Returns ([x, y, z], [vx, vy, vz]).
    pub fn state(
        &mut self,
        target: i32,
        center: i32,
        epoch: f64,
    ) -> Result<([f64; 3], [f64; 3]), SpkError> {
        let seg = self
            .segments
            .iter()
            .rev()
            .find(|s| {
                s.target == target
                    && s.center == center
                    && epoch >= s.start_epoch
                    && epoch <= s.end_epoch
            })
            .ok_or(SpkError::SegmentNotFound { target, center })?
            .clone();

        if seg.data_type != 2 {
            return Err(SpkError::UnsupportedType(seg.data_type));
        }

        evaluate_type2_state(&mut self.file, &self.header, &seg, epoch)
    }
}

// ── Segment enumeration ────────────────────────────────────────────

/// Read all segment descriptors from the DAF summary records.
fn read_all_segments(
    file: &mut File,
    header: &DafFileRecord,
) -> Result<Vec<SegmentDescriptor>, SpkError> {
    let mut segments = Vec::new();
    let mut record_num = header.fward;

    loop {
        if record_num == 0 {
            break;
        }

        // Read the summary record
        let record = read_record(file, record_num, header.endianness)?;

        // First 3 doubles are: next_record, prev_record, n_summaries
        let next_record = record[0] as usize;
        let n_summaries = record[2] as usize;

        // Parse each summary
        for i in 0..n_summaries {
            let offset = 3 + i * SPK_SS;
            if offset + SPK_SS > RECORD_DOUBLES {
                break;
            }

            let summary = &record[offset..offset + SPK_SS];
            let desc = unpack_summary(summary, header.endianness)?;
            segments.push(desc);
        }

        record_num = next_record;
    }

    Ok(segments)
}

/// Read a single 1024-byte record (1-indexed) as 128 f64 values.
fn read_record(
    file: &mut File,
    record_num: usize,
    endianness: Endianness,
) -> Result<[f64; RECORD_DOUBLES], SpkError> {
    let offset = (record_num as u64 - 1) * RECORD_BYTES;
    file.seek(SeekFrom::Start(offset))?;

    let mut buf = [0u8; RECORD_BYTES as usize];
    file.read_exact(&mut buf)?;

    let mut doubles = [0.0f64; RECORD_DOUBLES];
    for i in 0..RECORD_DOUBLES {
        let bytes: [u8; 8] = buf[i * 8..(i + 1) * 8].try_into().unwrap();
        doubles[i] = match endianness {
            Endianness::Little => f64::from_le_bytes(bytes),
            Endianness::Big => f64::from_be_bytes(bytes),
        };
    }

    Ok(doubles)
}

/// Unpack a 5-double SPK summary into a SegmentDescriptor.
///
/// Summary layout for SPK (ND=2, NI=6):
///   [0]: start epoch (f64)
///   [1]: end epoch (f64)
///   [2]: packed i32 pair (target, center)
///   [3]: packed i32 pair (frame, data_type)
///   [4]: packed i32 pair (start_addr, end_addr)
fn unpack_summary(summary: &[f64], endianness: Endianness) -> Result<SegmentDescriptor, SpkError> {
    let start_epoch = summary[0];
    let end_epoch = summary[1];

    // The integer components are packed as pairs into the remaining doubles.
    // Each f64 contains two i32 values (8 bytes = two 4-byte integers).
    let unpack_i32_pair = |val: f64| -> (i32, i32) {
        let bytes = match endianness {
            Endianness::Little => val.to_le_bytes(),
            Endianness::Big => val.to_be_bytes(),
        };
        let a = match endianness {
            Endianness::Little => i32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            Endianness::Big => i32::from_be_bytes(bytes[0..4].try_into().unwrap()),
        };
        let b = match endianness {
            Endianness::Little => i32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            Endianness::Big => i32::from_be_bytes(bytes[4..8].try_into().unwrap()),
        };
        (a, b)
    };

    let (target, center) = unpack_i32_pair(summary[2]);
    let (frame, data_type) = unpack_i32_pair(summary[3]);
    let (start_addr, end_addr) = unpack_i32_pair(summary[4]);

    Ok(SegmentDescriptor {
        start_epoch,
        end_epoch,
        target,
        center,
        frame,
        data_type,
        start_addr: start_addr as usize,
        end_addr: end_addr as usize,
    })
}

// ── Type 2 Chebyshev evaluation ────────────────────────────────────

/// Read the Type 2 segment directory (last 4 doubles of the segment data).
struct Type2Directory {
    /// Initial epoch of the first record (seconds past J2000 TDB).
    init: f64,
    /// Length of each record's time interval (seconds).
    intlen: f64,
    /// Number of doubles per record.
    rsize: usize,
    /// Number of records.
    n: usize,
}

/// Read the 4-double directory at the end of a Type 2 segment.
fn read_type2_directory(
    file: &mut File,
    header: &DafFileRecord,
    seg: &SegmentDescriptor,
) -> Result<Type2Directory, SpkError> {
    // The directory is the last 4 doubles of the segment data.
    // DAF addresses are 1-indexed. Each address holds one f64.
    // seg.end_addr points to the last double in the segment.
    // Directory occupies addresses: end_addr-3 .. end_addr

    let dir_start_addr = seg.end_addr - 3; // 1-indexed
    let byte_offset = (dir_start_addr as u64 - 1) * 8;

    file.seek(SeekFrom::Start(byte_offset))?;

    let mut buf = [0u8; 32]; // 4 doubles
    file.read_exact(&mut buf)?;

    let read_f64 = |i: usize| -> f64 {
        let bytes: [u8; 8] = buf[i * 8..(i + 1) * 8].try_into().unwrap();
        match header.endianness {
            Endianness::Little => f64::from_le_bytes(bytes),
            Endianness::Big => f64::from_be_bytes(bytes),
        }
    };

    Ok(Type2Directory {
        init: read_f64(0),
        intlen: read_f64(1),
        rsize: read_f64(2) as usize,
        n: read_f64(3) as usize,
    })
}

/// Read a single Type 2 Chebyshev record from the segment.
fn read_type2_record(
    file: &mut File,
    header: &DafFileRecord,
    seg: &SegmentDescriptor,
    dir: &Type2Directory,
    record_index: usize,
) -> Result<Vec<f64>, SpkError> {
    // Records start at seg.start_addr (1-indexed).
    // Record i starts at: start_addr + i * rsize
    let record_addr = seg.start_addr + record_index * dir.rsize; // 1-indexed
    let byte_offset = (record_addr as u64 - 1) * 8;

    file.seek(SeekFrom::Start(byte_offset))?;

    let n_bytes = dir.rsize * 8;
    let mut buf = vec![0u8; n_bytes];
    file.read_exact(&mut buf)?;

    let mut record = vec![0.0f64; dir.rsize];
    for i in 0..dir.rsize {
        let bytes: [u8; 8] = buf[i * 8..(i + 1) * 8].try_into().unwrap();
        record[i] = match header.endianness {
            Endianness::Little => f64::from_le_bytes(bytes),
            Endianness::Big => f64::from_be_bytes(bytes),
        };
    }

    Ok(record)
}

/// Evaluate a Chebyshev polynomial using Clenshaw's recurrence.
///
/// coeffs: Chebyshev coefficients [c0, c1, ..., cn]
/// t: normalized time in [-1, 1]
///
/// Returns (value, derivative w.r.t. t)
fn chebyshev_eval(coeffs: &[f64], t: f64) -> (f64, f64) {
    let n = coeffs.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    if n == 1 {
        return (coeffs[0], 0.0);
    }

    // Clenshaw recurrence for value: T_n(t)
    // S_{n+1} = 0, S_n = 0
    // S_i = 2*t*S_{i+1} - S_{i+2} + c_i
    // result = S_0 = c_0 + t*S_1 - S_2 (modified first step)
    //
    // Actually, standard Clenshaw for Chebyshev:
    // b_{n+1} = 0, b_{n} = 0
    // b_k = 2*t*b_{k+1} - b_{k+2} + c_k,  for k = n-1, n-2, ..., 1
    // result = c_0 + t*b_1 - b_2
    //
    // For derivative (dT_n/dt):
    // d_{n+1} = 0, d_n = 0
    // d_k = 2*b_{k+1} + 2*t*d_{k+1} - d_{k+2},  for k = n-1, ..., 1
    // result = b_1 + t*d_1 - d_2

    let mut b_k1 = 0.0; // b_{k+1}
    let mut b_k2 = 0.0; // b_{k+2}
    let mut d_k1 = 0.0; // d_{k+1}
    let mut d_k2 = 0.0; // d_{k+2}

    let two_t = 2.0 * t;

    for k in (1..n).rev() {
        let b_k = two_t * b_k1 - b_k2 + coeffs[k];
        let d_k = 2.0 * b_k1 + two_t * d_k1 - d_k2;

        b_k2 = b_k1;
        b_k1 = b_k;
        d_k2 = d_k1;
        d_k1 = d_k;
    }

    let value = coeffs[0] + t * b_k1 - b_k2;
    let deriv = b_k1 + t * d_k1 - d_k2;

    (value, deriv)
}

/// Evaluate Type 2 segment: position only.
fn evaluate_type2(
    file: &mut File,
    header: &DafFileRecord,
    seg: &SegmentDescriptor,
    epoch: f64,
) -> Result<[f64; 3], SpkError> {
    let dir = read_type2_directory(file, header, seg)?;

    // Determine which record covers this epoch
    let record_index = ((epoch - dir.init) / dir.intlen) as usize;
    let record_index = record_index.min(dir.n - 1);

    let record = read_type2_record(file, header, seg, &dir, record_index)?;

    // Record layout: [MID, RADIUS, X_coeffs..., Y_coeffs..., Z_coeffs...]
    let mid = record[0];
    let radius = record[1];

    // Number of coefficients per component
    let n_coeffs = (dir.rsize - 2) / 3;

    // Normalized time: map [MID-RADIUS, MID+RADIUS] -> [-1, 1]
    let t = (epoch - mid) / radius;

    let x_coeffs = &record[2..2 + n_coeffs];
    let y_coeffs = &record[2 + n_coeffs..2 + 2 * n_coeffs];
    let z_coeffs = &record[2 + 2 * n_coeffs..2 + 3 * n_coeffs];

    let (x, _) = chebyshev_eval(x_coeffs, t);
    let (y, _) = chebyshev_eval(y_coeffs, t);
    let (z, _) = chebyshev_eval(z_coeffs, t);

    Ok([x, y, z])
}

/// Evaluate Type 2 segment: position and velocity.
fn evaluate_type2_state(
    file: &mut File,
    header: &DafFileRecord,
    seg: &SegmentDescriptor,
    epoch: f64,
) -> Result<([f64; 3], [f64; 3]), SpkError> {
    let dir = read_type2_directory(file, header, seg)?;

    let record_index = ((epoch - dir.init) / dir.intlen) as usize;
    let record_index = record_index.min(dir.n - 1);

    let record = read_type2_record(file, header, seg, &dir, record_index)?;

    let mid = record[0];
    let radius = record[1];
    let n_coeffs = (dir.rsize - 2) / 3;
    let t = (epoch - mid) / radius;

    let x_coeffs = &record[2..2 + n_coeffs];
    let y_coeffs = &record[2 + n_coeffs..2 + 2 * n_coeffs];
    let z_coeffs = &record[2 + 2 * n_coeffs..2 + 3 * n_coeffs];

    let (x, dx_dt) = chebyshev_eval(x_coeffs, t);
    let (y, dy_dt) = chebyshev_eval(y_coeffs, t);
    let (z, dz_dt) = chebyshev_eval(z_coeffs, t);

    // Chain rule: dX/dt_real = dX/dt_normalized * dt_normalized/dt_real = dX/dt / radius
    let vx = dx_dt / radius;
    let vy = dy_dt / radius;
    let vz = dz_dt / radius;

    Ok(([x, y, z], [vx, vy, vz]))
}

// ── Multi-body position chaining ───────────────────────────────────

impl SpkFile {
    /// Compute position of `target` relative to `observer` by chaining segments.
    ///
    /// For DE440, the Sun (10) position relative to Earth (399) requires:
    ///   Earth (399) -> Earth-Moon Barycenter (3): segment target=399, center=3
    ///   Earth-Moon Barycenter (3) -> SSB (0): segment target=3, center=0
    ///   Sun (10) -> SSB (0): segment target=10, center=0
    ///
    /// sun_wrt_earth = sun_wrt_ssb - earth_wrt_ssb
    ///               = sun_wrt_ssb - (emb_wrt_ssb + earth_wrt_emb)
    pub fn position_chain(
        &mut self,
        target: i32,
        observer: i32,
        epoch: f64,
    ) -> Result<[f64; 3], SpkError> {
        // Try direct lookup first
        if let Ok(pos) = self.position(target, observer, epoch) {
            return Ok(pos);
        }

        // Chain through SSB (body 0)
        let target_wrt_ssb = self.position_relative_to_ssb(target, epoch)?;
        let observer_wrt_ssb = self.position_relative_to_ssb(observer, epoch)?;

        Ok([
            target_wrt_ssb[0] - observer_wrt_ssb[0],
            target_wrt_ssb[1] - observer_wrt_ssb[1],
            target_wrt_ssb[2] - observer_wrt_ssb[2],
        ])
    }

    /// Get position of a body relative to SSB by chaining available segments.
    fn position_relative_to_ssb(&mut self, body: i32, epoch: f64) -> Result<[f64; 3], SpkError> {
        let mut current = body;
        let mut total = [0.0f64; 3];

        // Walk up the chain until we reach SSB (0)
        let mut iterations = 0;
        while current != SOLAR_SYSTEM_BARYCENTER {
            if iterations > 10 {
                return Err(SpkError::SegmentNotFound {
                    target: body,
                    center: SOLAR_SYSTEM_BARYCENTER,
                });
            }

            // Find any segment with this body as target
            let seg = self
                .segments
                .iter()
                .rev()
                .find(|s| s.target == current && epoch >= s.start_epoch && epoch <= s.end_epoch)
                .ok_or(SpkError::SegmentNotFound {
                    target: current,
                    center: SOLAR_SYSTEM_BARYCENTER,
                })?
                .clone();

            if seg.data_type != 2 {
                return Err(SpkError::UnsupportedType(seg.data_type));
            }

            let pos = evaluate_type2(&mut self.file, &self.header, &seg, epoch)?;
            total[0] += pos[0];
            total[1] += pos[1];
            total[2] += pos[2];

            current = seg.center;
            iterations += 1;
        }

        Ok(total)
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chebyshev_eval_constant() {
        // f(t) = 5.0 (constant)
        let coeffs = [5.0];
        let (val, deriv) = chebyshev_eval(&coeffs, 0.0);
        assert!((val - 5.0).abs() < 1e-15);
        assert!(deriv.abs() < 1e-15);
    }

    #[test]
    fn test_chebyshev_eval_linear() {
        // f(t) = c0 + c1*T1(t) = c0 + c1*t
        // With c0=3.0, c1=2.0: f(t) = 3 + 2t
        let coeffs = [3.0, 2.0];

        let (val, deriv) = chebyshev_eval(&coeffs, 0.0);
        assert!((val - 3.0).abs() < 1e-15);
        assert!((deriv - 2.0).abs() < 1e-15);

        let (val, deriv) = chebyshev_eval(&coeffs, 1.0);
        assert!((val - 5.0).abs() < 1e-15);
        assert!((deriv - 2.0).abs() < 1e-15);

        let (val, deriv) = chebyshev_eval(&coeffs, -1.0);
        assert!((val - 1.0).abs() < 1e-15);
        assert!((deriv - 2.0).abs() < 1e-15);
    }

    #[test]
    fn test_chebyshev_eval_quadratic() {
        // f(t) = c0 + c1*T1(t) + c2*T2(t)
        // T2(t) = 2t^2 - 1
        // With c0=1.0, c1=0.0, c2=1.0: f(t) = 1 + (2t^2 - 1) = 2t^2
        let coeffs = [1.0, 0.0, 1.0];

        let (val, _) = chebyshev_eval(&coeffs, 0.0);
        assert!((val - 0.0).abs() < 1e-15, "f(0) = {}", val);

        let (val, _) = chebyshev_eval(&coeffs, 1.0);
        assert!((val - 2.0).abs() < 1e-15, "f(1) = {}", val);

        let (val, _) = chebyshev_eval(&coeffs, 0.5);
        // f(0.5) = 2*(0.25) = 0.5
        assert!((val - 0.5).abs() < 1e-15, "f(0.5) = {}", val);
    }

    #[test]
    fn test_chebyshev_derivative_quadratic() {
        // f(t) = 2t^2, f'(t) = 4t
        // Chebyshev: c0=1, c1=0, c2=1
        let coeffs = [1.0, 0.0, 1.0];

        let (_, deriv) = chebyshev_eval(&coeffs, 0.5);
        // f'(0.5) = 4*0.5 = 2.0
        assert!((deriv - 2.0).abs() < 1e-14, "f'(0.5) = {}", deriv);

        let (_, deriv) = chebyshev_eval(&coeffs, 0.0);
        assert!(deriv.abs() < 1e-15, "f'(0) = {}", deriv);
    }

    #[test]
    fn test_chebyshev_eval_cubic() {
        // T3(t) = 4t^3 - 3t
        // f(t) = c0 + c1*T1 + c2*T2 + c3*T3
        // With c0=0, c1=0, c2=0, c3=1: f(t) = 4t^3 - 3t
        let coeffs = [0.0, 0.0, 0.0, 1.0];

        let (val, _) = chebyshev_eval(&coeffs, 1.0);
        // f(1) = 4 - 3 = 1
        assert!((val - 1.0).abs() < 1e-14, "f(1) = {}", val);

        let (val, _) = chebyshev_eval(&coeffs, 0.5);
        // f(0.5) = 4*0.125 - 1.5 = 0.5 - 1.5 = -1.0
        assert!((val - (-1.0)).abs() < 1e-14, "f(0.5) = {}", val);
    }

    // Integration test: only runs if DE440 file is available
    #[test]
    #[ignore] // Run with: cargo test --release -- --ignored
    fn test_read_de440() {
        let path = std::env::var("DE440_PATH")
            .unwrap_or_else(|_| "/Users/mostafamahdi/twilight/data/de440.bsp".to_string());

        if !std::path::Path::new(&path).exists() {
            eprintln!("DE440 file not found at {}. Skipping test.", path);
            return;
        }

        let spk = SpkFile::open(&path).expect("failed to open DE440");
        println!("DE440 loaded: {} segments", spk.segments().len());

        for seg in spk.segments() {
            println!(
                "  target={:>4} center={:>4} frame={} type={} epochs=[{:.1}, {:.1}]",
                seg.target, seg.center, seg.frame, seg.data_type, seg.start_epoch, seg.end_epoch
            );
        }
    }

    #[test]
    #[ignore]
    fn test_de440_sun_position() {
        let path = std::env::var("DE440_PATH")
            .unwrap_or_else(|_| "/Users/mostafamahdi/twilight/data/de440.bsp".to_string());

        if !std::path::Path::new(&path).exists() {
            eprintln!("DE440 file not found. Skipping.");
            return;
        }

        let mut spk = SpkFile::open(&path).expect("failed to open DE440");

        // J2000.0 epoch = 0.0 seconds past J2000 TDB
        let epoch = 0.0;
        let sun_pos = spk
            .position_chain(SUN, EARTH, epoch)
            .expect("failed to get Sun position");

        println!(
            "Sun position wrt Earth at J2000.0: [{:.6}, {:.6}, {:.6}] km",
            sun_pos[0], sun_pos[1], sun_pos[2]
        );

        // Verify the distance is approximately 1 AU (149,597,870.7 km)
        let dist =
            (sun_pos[0] * sun_pos[0] + sun_pos[1] * sun_pos[1] + sun_pos[2] * sun_pos[2]).sqrt();
        println!("Distance: {:.3} km ({:.6} AU)", dist, dist / 149_597_870.7);

        // Should be within ~3% of 1 AU (Earth is slightly elliptical)
        assert!(
            dist > 145_000_000.0 && dist < 155_000_000.0,
            "distance {} km is not near 1 AU",
            dist
        );
    }
}
