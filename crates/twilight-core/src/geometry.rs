//! Ray-sphere intersection, coordinate transforms, and 3D vector math.

use libm::sqrt;

/// 3D vector for position and direction calculations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    #[inline]
    pub fn length(self) -> f64 {
        sqrt(self.dot(self))
    }

    #[inline]
    pub fn length_sq(self) -> f64 {
        self.dot(self)
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-15 {
            return Self::new(0.0, 0.0, 0.0);
        }
        self * (1.0 / len)
    }

    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}

impl core::ops::Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl core::ops::Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl core::ops::Mul<f64> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

impl core::ops::Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

/// Result of a ray-sphere intersection test.
#[derive(Debug, Clone, Copy)]
pub struct RaySphereHit {
    /// Distance to the near intersection (may be negative if inside sphere).
    pub t_near: f64,
    /// Distance to the far intersection.
    pub t_far: f64,
}

/// Compute ray-sphere intersection for a sphere centered at origin with given radius.
///
/// Ray: P(t) = origin + t * direction
/// Sphere: |P|^2 = radius^2
///
/// Returns None if the ray misses the sphere entirely.
/// If the ray origin is inside the sphere, t_near will be negative.
pub fn ray_sphere_intersect(origin: Vec3, direction: Vec3, radius: f64) -> Option<RaySphereHit> {
    // Solve: |origin + t*dir|^2 = radius^2
    // => t^2*(dir·dir) + 2t*(origin·dir) + (origin·origin - radius^2) = 0
    let a = direction.dot(direction);
    let b = 2.0 * origin.dot(direction);
    let c = origin.dot(origin) - radius * radius;

    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        return None;
    }

    let sqrt_disc = sqrt(discriminant);
    let inv_2a = 0.5 / a;

    // Use numerically stable formula to avoid catastrophic cancellation
    let t_near = (-b - sqrt_disc) * inv_2a;
    let t_far = (-b + sqrt_disc) * inv_2a;

    Some(RaySphereHit { t_near, t_far })
}

/// Find intersection distance to the next atmospheric shell boundary.
///
/// Given a photon at `position` traveling in `direction`, find the distance
/// to either the inner or outer shell boundary. The photon is between two
/// concentric spheres of radii `r_inner` and `r_outer`.
///
/// Returns (distance, is_outward) where is_outward indicates if the photon
/// hits the outer boundary (true) or inner boundary (false).
pub fn next_shell_boundary(
    position: Vec3,
    direction: Vec3,
    r_inner: f64,
    r_outer: f64,
) -> Option<(f64, bool)> {
    let _r = position.length();

    // Try outer boundary first
    if let Some(hit) = ray_sphere_intersect(position, direction, r_outer) {
        // We want the nearest positive intersection
        if hit.t_near > 1e-10 {
            // Check if inner boundary is closer
            if let Some(inner_hit) = ray_sphere_intersect(position, direction, r_inner) {
                if inner_hit.t_near > 1e-10 && inner_hit.t_near < hit.t_near {
                    return Some((inner_hit.t_near, false));
                }
            }
            return Some((hit.t_near, true));
        }
        if hit.t_far > 1e-10 {
            // Inside outer sphere, check inner
            if let Some(inner_hit) = ray_sphere_intersect(position, direction, r_inner) {
                if inner_hit.t_near > 1e-10 && inner_hit.t_near < hit.t_far {
                    return Some((inner_hit.t_near, false));
                }
            }
            return Some((hit.t_far, true));
        }
    }

    // Fallback: check inner boundary only
    if let Some(inner_hit) = ray_sphere_intersect(position, direction, r_inner) {
        if inner_hit.t_near > 1e-10 {
            return Some((inner_hit.t_near, false));
        }
    }

    None
}

// ── Atmospheric refraction at shell boundaries ─────────────────────────

/// Result of applying Snell's law at a spherical shell boundary.
#[derive(Debug, Clone, Copy)]
pub enum RefractResult {
    /// Ray was refracted into the new medium. Contains the new unit direction.
    Refracted(Vec3),
    /// Total internal reflection occurred (grazing ray from denser to rarer
    /// medium). Contains the reflected unit direction. The photon remains
    /// in the original shell.
    TotalReflection(Vec3),
}

/// Apply Snell's law at a concentric spherical shell boundary.
///
/// Refracts (or reflects) a ray crossing from a medium with refractive
/// index `n_from` to one with `n_to`. The boundary is a sphere centered
/// at the origin; `boundary_pos` is the exact position on that sphere.
///
/// # Fast path
///
/// When `n_from == n_to` (within 1e-15), this returns the original
/// direction unchanged. This means that when the atmosphere model has
/// all refractive indices set to 1.0 (the default), every boundary
/// crossing is a no-op and the transport is identical to straight-line.
///
/// # Physics
///
/// For a spherically symmetric atmosphere with piecewise-constant n,
/// Bouger's invariant n*r*sin(alpha) = const reduces to standard
/// Snell's law at each shell boundary because the surface normal is
/// the radial direction.
pub fn refract_at_boundary(dir: Vec3, boundary_pos: Vec3, n_from: f64, n_to: f64) -> RefractResult {
    // Fast path: no refraction when indices match (includes n=1 everywhere).
    if libm::fabs(n_from - n_to) < 1e-15 {
        return RefractResult::Refracted(dir);
    }

    let outward_normal = boundary_pos.normalize();

    // The interface normal must point toward the side the ray is arriving
    // from (standard Snell convention). Determine by checking whether the
    // ray is traveling inward or outward relative to the sphere.
    let cos_dir_normal = dir.dot(outward_normal);
    let normal = if cos_dir_normal < 0.0 {
        outward_normal // ray going inward: normal points outward (toward incoming side)
    } else {
        -outward_normal // ray going outward: normal points inward (toward incoming side)
    };

    let cos_i = -(dir.dot(normal)); // always >= 0 by construction
    let eta = n_from / n_to;
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);

    if k < 0.0 {
        // Total internal reflection (possible for extremely grazing rays
        // going from denser to rarer medium -- creates atmospheric ducts,
        // the mechanism behind mirages).
        let reflected = dir + normal * (2.0 * cos_i);
        return RefractResult::TotalReflection(reflected.normalize());
    }

    let cos_t = sqrt(k);
    let refracted = dir * eta + normal * (eta * cos_i - cos_t);
    RefractResult::Refracted(refracted.normalize())
}

/// Convert geographic coordinates (latitude, longitude, altitude) to
/// Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates.
///
/// - `lat_deg`: latitude in degrees (north positive)
/// - `lon_deg`: longitude in degrees (east positive)
/// - `altitude_m`: altitude above sea level in meters
///
/// Uses spherical Earth with mean radius 6371 km.
pub fn geographic_to_ecef(lat_deg: f64, lon_deg: f64, altitude_m: f64) -> Vec3 {
    use libm::{cos, sin};

    const EARTH_RADIUS_M: f64 = 6_371_000.0;
    let lat = lat_deg * core::f64::consts::PI / 180.0;
    let lon = lon_deg * core::f64::consts::PI / 180.0;
    let r = EARTH_RADIUS_M + altitude_m;

    Vec3::new(
        r * cos(lat) * cos(lon),
        r * cos(lat) * sin(lon),
        r * sin(lat),
    )
}

/// Compute the solar direction vector in ECEF coordinates given
/// solar zenith angle (SZA) and solar azimuth angle (SAA) at a given
/// observer location.
///
/// - `sza_deg`: solar zenith angle in degrees (0 = overhead, 90 = horizon)
/// - `saa_deg`: solar azimuth angle in degrees (0 = north, clockwise)
/// - `lat_deg`: observer latitude in degrees
/// - `lon_deg`: observer longitude in degrees
///
/// Returns a unit vector pointing from the observer toward the sun.
pub fn solar_direction_ecef(sza_deg: f64, saa_deg: f64, lat_deg: f64, lon_deg: f64) -> Vec3 {
    use libm::{cos, sin};

    let sza = sza_deg * core::f64::consts::PI / 180.0;
    let saa = saa_deg * core::f64::consts::PI / 180.0;
    let lat = lat_deg * core::f64::consts::PI / 180.0;
    let lon = lon_deg * core::f64::consts::PI / 180.0;

    // Solar direction in local ENU (East-North-Up)
    let sun_e = sin(sza) * sin(saa);
    let sun_n = sin(sza) * cos(saa);
    let sun_u = cos(sza);

    // ENU to ECEF rotation
    // East  = (-sin(lon), cos(lon), 0)
    // North = (-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat))
    // Up    = (cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat))
    Vec3::new(
        -sin(lon) * sun_e - sin(lat) * cos(lon) * sun_n + cos(lat) * cos(lon) * sun_u,
        cos(lon) * sun_e - sin(lat) * sin(lon) * sun_n + cos(lat) * sin(lon) * sun_u,
        cos(lat) * sun_n + sin(lat) * sun_u,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-12;
    const EPSILON_GEO: f64 = 1e-6; // for trig-based functions

    // ── Vec3 basic operations ──

    #[test]
    fn vec3_new_stores_components() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn vec3_dot_orthogonal_is_zero() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert!((a.dot(b)).abs() < EPSILON);
    }

    #[test]
    fn vec3_dot_parallel_is_product_of_lengths() {
        let a = Vec3::new(3.0, 0.0, 0.0);
        let b = Vec3::new(5.0, 0.0, 0.0);
        assert!((a.dot(b) - 15.0).abs() < EPSILON);
    }

    #[test]
    fn vec3_dot_antiparallel_is_negative() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(-1.0, 0.0, 0.0);
        assert!((a.dot(b) - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn vec3_dot_is_commutative() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!((a.dot(b) - b.dot(a)).abs() < EPSILON);
    }

    #[test]
    fn vec3_dot_known_value() {
        // (1,2,3)·(4,5,6) = 4+10+18 = 32
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!((a.dot(b) - 32.0).abs() < EPSILON);
    }

    #[test]
    fn vec3_cross_orthogonal_basis() {
        // x × y = z
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = x.cross(y);
        assert!((z.x - 0.0).abs() < EPSILON);
        assert!((z.y - 0.0).abs() < EPSILON);
        assert!((z.z - 1.0).abs() < EPSILON);
    }

    #[test]
    fn vec3_cross_anticommutative() {
        // a × b = -(b × a)
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let ab = a.cross(b);
        let ba = b.cross(a);
        assert!((ab.x + ba.x).abs() < EPSILON);
        assert!((ab.y + ba.y).abs() < EPSILON);
        assert!((ab.z + ba.z).abs() < EPSILON);
    }

    #[test]
    fn vec3_cross_self_is_zero() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let c = a.cross(a);
        assert!(c.length() < EPSILON);
    }

    #[test]
    fn vec3_cross_known_value() {
        // (1,2,3) × (4,5,6) = (2*6-3*5, 3*4-1*6, 1*5-2*4) = (-3, 6, -3)
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let c = a.cross(b);
        assert!((c.x - (-3.0)).abs() < EPSILON);
        assert!((c.y - 6.0).abs() < EPSILON);
        assert!((c.z - (-3.0)).abs() < EPSILON);
    }

    #[test]
    fn vec3_length_345_triangle() {
        // Classic 3-4-5 right triangle
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!((v.length() - 5.0).abs() < EPSILON);
    }

    #[test]
    fn vec3_length_unit_vectors() {
        assert!((Vec3::new(1.0, 0.0, 0.0).length() - 1.0).abs() < EPSILON);
        assert!((Vec3::new(0.0, 1.0, 0.0).length() - 1.0).abs() < EPSILON);
        assert!((Vec3::new(0.0, 0.0, 1.0).length() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn vec3_length_zero_vector() {
        assert!((Vec3::new(0.0, 0.0, 0.0).length()).abs() < EPSILON);
    }

    #[test]
    fn vec3_length_sq_is_length_squared() {
        let v = Vec3::new(3.0, 4.0, 5.0);
        assert!((v.length_sq() - v.length() * v.length()).abs() < EPSILON);
    }

    #[test]
    fn vec3_normalize_produces_unit_vector() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < EPSILON);
        assert!((n.x - 0.6).abs() < EPSILON);
        assert!((n.y - 0.8).abs() < EPSILON);
    }

    #[test]
    fn vec3_normalize_zero_returns_zero() {
        let v = Vec3::new(0.0, 0.0, 0.0);
        let n = v.normalize();
        assert!(n.length() < EPSILON);
    }

    #[test]
    fn vec3_normalize_preserves_direction() {
        let v = Vec3::new(7.0, -3.0, 11.0);
        let n = v.normalize();
        // n should be parallel to v: n × v = 0
        let cross = n.cross(v);
        assert!(cross.length() < 1e-10);
    }

    #[test]
    fn vec3_scale() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let s = v.scale(2.0);
        assert!((s.x - 2.0).abs() < EPSILON);
        assert!((s.y - 4.0).abs() < EPSILON);
        assert!((s.z - 6.0).abs() < EPSILON);
    }

    #[test]
    fn vec3_add() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let c = a + b;
        assert!((c.x - 5.0).abs() < EPSILON);
        assert!((c.y - 7.0).abs() < EPSILON);
        assert!((c.z - 9.0).abs() < EPSILON);
    }

    #[test]
    fn vec3_sub() {
        let a = Vec3::new(4.0, 5.0, 6.0);
        let b = Vec3::new(1.0, 2.0, 3.0);
        let c = a - b;
        assert!((c.x - 3.0).abs() < EPSILON);
        assert!((c.y - 3.0).abs() < EPSILON);
        assert!((c.z - 3.0).abs() < EPSILON);
    }

    #[test]
    fn vec3_mul_scalar() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let s = v * 3.0;
        assert!((s.x - 3.0).abs() < EPSILON);
        assert!((s.y - 6.0).abs() < EPSILON);
        assert!((s.z - 9.0).abs() < EPSILON);
    }

    #[test]
    fn vec3_neg() {
        let v = Vec3::new(1.0, -2.0, 3.0);
        let n = -v;
        assert!((n.x - (-1.0)).abs() < EPSILON);
        assert!((n.y - 2.0).abs() < EPSILON);
        assert!((n.z - (-3.0)).abs() < EPSILON);
    }

    // ── ray_sphere_intersect ──

    #[test]
    fn ray_sphere_miss() {
        // Ray from (0, 2, 0) going in +x, sphere of radius 1 at origin → miss
        let origin = Vec3::new(0.0, 2.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        assert!(ray_sphere_intersect(origin, dir, 1.0).is_none());
    }

    #[test]
    fn ray_sphere_hit_along_x_axis() {
        // Ray from (-10, 0, 0) going in +x, sphere radius 1
        // Should hit at t=9 and t=11
        let origin = Vec3::new(-10.0, 0.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let hit = ray_sphere_intersect(origin, dir, 1.0).unwrap();
        assert!((hit.t_near - 9.0).abs() < EPSILON);
        assert!((hit.t_far - 11.0).abs() < EPSILON);
    }

    #[test]
    fn ray_sphere_from_inside() {
        // Origin inside sphere → t_near < 0, t_far > 0
        let origin = Vec3::new(0.5, 0.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let hit = ray_sphere_intersect(origin, dir, 1.0).unwrap();
        assert!(hit.t_near < 0.0);
        assert!(hit.t_far > 0.0);
    }

    #[test]
    fn ray_sphere_tangent() {
        // Ray tangent to unit sphere: origin at (0, 1, 0), direction +x
        // Discriminant = 0, t_near = t_far = 0
        let origin = Vec3::new(0.0, 1.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let hit = ray_sphere_intersect(origin, dir, 1.0).unwrap();
        assert!((hit.t_near - hit.t_far).abs() < 1e-6);
    }

    #[test]
    fn ray_sphere_from_origin_outward() {
        // From (0,0,0) looking +x at sphere radius R
        // Should hit at t = -R (behind) and t = +R (in front)
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let r = 100.0;
        let hit = ray_sphere_intersect(origin, dir, r).unwrap();
        assert!((hit.t_near - (-r)).abs() < EPSILON);
        assert!((hit.t_far - r).abs() < EPSILON);
    }

    #[test]
    fn ray_sphere_large_radius_earth() {
        // Observer on Earth's surface looking up: origin at (R_earth, 0, 0), dir = (+1,0,0)
        // Intersect with sphere of radius R_earth + 100km
        let r_earth = 6_371_000.0;
        let r_toa = r_earth + 100_000.0;
        let origin = Vec3::new(r_earth, 0.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let hit = ray_sphere_intersect(origin, dir, r_toa).unwrap();
        // t_near should be negative (behind), t_far should be +100km
        assert!(hit.t_near < 0.0);
        assert!((hit.t_far - 100_000.0).abs() < 1.0); // ~1m tolerance
    }

    // ── next_shell_boundary ──

    #[test]
    fn next_shell_boundary_outward() {
        // Photon at radius 5, between shells r_inner=4, r_outer=6, going outward radially
        let pos = Vec3::new(5.0, 0.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let result = next_shell_boundary(pos, dir, 4.0, 6.0);
        assert!(result.is_some());
        let (dist, is_outward) = result.unwrap();
        assert!(is_outward);
        assert!((dist - 1.0).abs() < 1e-6); // 5 → 6 = 1 unit
    }

    #[test]
    fn next_shell_boundary_inward() {
        // Photon at radius 5, going inward (toward -x)
        let pos = Vec3::new(5.0, 0.0, 0.0);
        let dir = Vec3::new(-1.0, 0.0, 0.0);
        let result = next_shell_boundary(pos, dir, 4.0, 6.0);
        assert!(result.is_some());
        let (dist, is_outward) = result.unwrap();
        assert!(!is_outward); // hits inner boundary
        assert!((dist - 1.0).abs() < 1e-6); // 5 → 4 = 1 unit
    }

    // ── geographic_to_ecef ──

    #[test]
    fn ecef_equator_prime_meridian() {
        // (0°N, 0°E, 0m) → (R, 0, 0)
        let r_earth = 6_371_000.0;
        let pos = geographic_to_ecef(0.0, 0.0, 0.0);
        assert!((pos.x - r_earth).abs() < 1.0);
        assert!(pos.y.abs() < 1.0);
        assert!(pos.z.abs() < 1.0);
    }

    #[test]
    fn ecef_north_pole() {
        // (90°N, 0°E, 0m) → (0, 0, R)
        let r_earth = 6_371_000.0;
        let pos = geographic_to_ecef(90.0, 0.0, 0.0);
        assert!(pos.x.abs() < 1.0);
        assert!(pos.y.abs() < 1.0);
        assert!((pos.z - r_earth).abs() < 1.0);
    }

    #[test]
    fn ecef_south_pole() {
        // (-90°N, 0°E, 0m) → (0, 0, -R)
        let r_earth = 6_371_000.0;
        let pos = geographic_to_ecef(-90.0, 0.0, 0.0);
        assert!(pos.x.abs() < 1.0);
        assert!(pos.y.abs() < 1.0);
        assert!((pos.z - (-r_earth)).abs() < 1.0);
    }

    #[test]
    fn ecef_equator_90e() {
        // (0°N, 90°E, 0m) → (0, R, 0)
        let r_earth = 6_371_000.0;
        let pos = geographic_to_ecef(0.0, 90.0, 0.0);
        assert!(pos.x.abs() < 1.0);
        assert!((pos.y - r_earth).abs() < 1.0);
        assert!(pos.z.abs() < 1.0);
    }

    #[test]
    fn ecef_altitude_adds_to_radius() {
        let alt = 1000.0; // 1 km
        let pos_surface = geographic_to_ecef(0.0, 0.0, 0.0);
        let pos_alt = geographic_to_ecef(0.0, 0.0, alt);
        let delta = pos_alt.length() - pos_surface.length();
        assert!((delta - alt).abs() < 0.1);
    }

    #[test]
    fn ecef_all_points_have_correct_radius() {
        // Any point should be at distance R + altitude from origin
        let r_earth = 6_371_000.0;
        let alt = 500.0;
        for lat in &[-90.0, -45.0, 0.0, 30.0, 60.0, 90.0] {
            for lon in &[-180.0, -90.0, 0.0, 90.0, 180.0] {
                let pos = geographic_to_ecef(*lat, *lon, alt);
                let expected_r = r_earth + alt;
                assert!(
                    (pos.length() - expected_r).abs() < 1.0,
                    "Failed at lat={}, lon={}: got r={}, expected {}",
                    lat,
                    lon,
                    pos.length(),
                    expected_r
                );
            }
        }
    }

    // ── solar_direction_ecef ──

    #[test]
    fn solar_direction_overhead_at_equator() {
        // SZA=0 (sun directly overhead) at (0°N, 0°E)
        // Sun direction should be straight up = radial direction = (1,0,0) at equator/prime meridian
        let dir = solar_direction_ecef(0.0, 0.0, 0.0, 0.0);
        assert!(
            (dir.length() - 1.0).abs() < EPSILON_GEO,
            "Should be unit vector"
        );
        // At (0°N, 0°E), up = (1, 0, 0) in ECEF
        assert!((dir.x - 1.0).abs() < EPSILON_GEO);
        assert!(dir.y.abs() < EPSILON_GEO);
        assert!(dir.z.abs() < EPSILON_GEO);
    }

    #[test]
    fn solar_direction_is_unit_vector() {
        // For various SZA/SAA/lat/lon, result should always be unit length
        let cases = [
            (45.0, 180.0, 30.0, 45.0),
            (96.0, 270.0, 21.4, 39.8),
            (108.0, 90.0, 51.5, -0.1),
        ];
        for (sza, saa, lat, lon) in &cases {
            let dir = solar_direction_ecef(*sza, *saa, *lat, *lon);
            assert!(
                (dir.length() - 1.0).abs() < EPSILON_GEO,
                "Not unit: sza={}, saa={}, lat={}, lon={}, len={}",
                sza,
                saa,
                lat,
                lon,
                dir.length()
            );
        }
    }

    #[test]
    fn solar_direction_horizon_perpendicular_to_up() {
        // SZA=90 means sun on horizon → sun_dir · up = cos(90°) = 0
        // At (0°N, 0°E), up = (1,0,0)
        let dir = solar_direction_ecef(90.0, 0.0, 0.0, 0.0);
        let up = Vec3::new(1.0, 0.0, 0.0);
        assert!(
            dir.dot(up).abs() < EPSILON_GEO,
            "Horizon sun should be perpendicular to up, got dot={}",
            dir.dot(up)
        );
    }

    #[test]
    fn solar_direction_sza_equals_angle_to_zenith() {
        // The dot product of sun_dir with the local up vector should equal cos(SZA)
        // At (0°N, 0°E), local up = (1, 0, 0)
        for sza_deg in &[0.0, 30.0, 60.0, 90.0, 108.0, 120.0] {
            let dir = solar_direction_ecef(*sza_deg, 180.0, 0.0, 0.0);
            let up = Vec3::new(1.0, 0.0, 0.0);
            let cos_sza = libm::cos(*sza_deg * core::f64::consts::PI / 180.0);
            assert!(
                (dir.dot(up) - cos_sza).abs() < EPSILON_GEO,
                "SZA={}: dot={}, expected cos(SZA)={}",
                sza_deg,
                dir.dot(up),
                cos_sza
            );
        }
    }

    // ── refract_at_boundary ──

    #[test]
    fn refract_identity_when_indices_equal() {
        // n_from == n_to: direction should be unchanged
        let dir = Vec3::new(0.3, -0.9, 0.1).normalize();
        let boundary_pos = Vec3::new(100.0, 0.0, 0.0); // on a sphere of r=100
        match refract_at_boundary(dir, boundary_pos, 1.000293, 1.000293) {
            RefractResult::Refracted(d) => {
                assert!((d.x - dir.x).abs() < 1e-14);
                assert!((d.y - dir.y).abs() < 1e-14);
                assert!((d.z - dir.z).abs() < 1e-14);
            }
            RefractResult::TotalReflection(_) => {
                panic!("Should not get TIR when n_from == n_to");
            }
        }
    }

    #[test]
    fn refract_identity_when_both_one() {
        // The default case: n=1.0 everywhere. Fast path.
        let dir = Vec3::new(0.0, 0.5, -0.866).normalize();
        let boundary_pos = Vec3::new(6_371_100.0, 0.0, 0.0);
        match refract_at_boundary(dir, boundary_pos, 1.0, 1.0) {
            RefractResult::Refracted(d) => {
                assert!((d.x - dir.x).abs() < 1e-14);
                assert!((d.y - dir.y).abs() < 1e-14);
                assert!((d.z - dir.z).abs() < 1e-14);
            }
            RefractResult::TotalReflection(_) => {
                panic!("Should not get TIR when n=1");
            }
        }
    }

    #[test]
    fn refract_normal_incidence_no_deflection() {
        // A ray traveling exactly radially should not deflect at any n ratio.
        // Outward ray on a sphere at r=100, boundary at (100,0,0).
        let dir = Vec3::new(1.0, 0.0, 0.0); // radially outward
        let boundary_pos = Vec3::new(100.0, 0.0, 0.0);
        let n_from = 1.000293;
        let n_to = 1.000200;
        match refract_at_boundary(dir, boundary_pos, n_from, n_to) {
            RefractResult::Refracted(d) => {
                // At normal incidence, sin(theta_i) = 0, so sin(theta_t) = 0.
                // The refracted direction should be the same as the incident.
                assert!(
                    (d.x - dir.x).abs() < 1e-10,
                    "Normal incidence should not deflect: d={:?}",
                    d
                );
                assert!(d.y.abs() < 1e-10);
                assert!(d.z.abs() < 1e-10);
            }
            RefractResult::TotalReflection(_) => {
                panic!("Normal incidence should never produce TIR");
            }
        }
    }

    #[test]
    fn refract_normal_incidence_inward() {
        // Inward radial ray should also pass through undeflected
        let dir = Vec3::new(-1.0, 0.0, 0.0); // radially inward
        let boundary_pos = Vec3::new(100.0, 0.0, 0.0);
        match refract_at_boundary(dir, boundary_pos, 1.000293, 1.000100) {
            RefractResult::Refracted(d) => {
                assert!(
                    (d.x - (-1.0)).abs() < 1e-10,
                    "Inward normal incidence should not deflect"
                );
                assert!(d.y.abs() < 1e-10);
                assert!(d.z.abs() < 1e-10);
            }
            RefractResult::TotalReflection(_) => {
                panic!("Normal incidence should never produce TIR");
            }
        }
    }

    #[test]
    fn refract_output_is_unit_vector() {
        // For various directions and n ratios, output must be unit length.
        let boundary_pos = Vec3::new(6_371_000.0 + 10_000.0, 0.0, 0.0);
        let dirs = [
            Vec3::new(0.3, -0.9, 0.1),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.99, -0.1, 0.0),
            Vec3::new(-0.5, -0.5, 0.707),
        ];
        let n_pairs = [(1.000293, 1.000200), (1.000200, 1.000293), (1.0003, 1.0)];

        for d in &dirs {
            let dir = d.normalize();
            for &(n_from, n_to) in &n_pairs {
                match refract_at_boundary(dir, boundary_pos, n_from, n_to) {
                    RefractResult::Refracted(r) | RefractResult::TotalReflection(r) => {
                        assert!(
                            (r.length() - 1.0).abs() < 1e-10,
                            "Output must be unit: |d|={}, n={}/{}",
                            r.length(),
                            n_from,
                            n_to
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn refract_snell_law_sin_ratio() {
        // Verify sin(theta_t)/sin(theta_i) = n_from/n_to for a known case.
        // Ray in the x-y plane hitting a sphere at (R, 0, 0).
        let r = 100.0;
        let boundary_pos = Vec3::new(r, 0.0, 0.0);
        let normal = boundary_pos.normalize(); // (1, 0, 0)

        // Incident angle ~30 degrees from normal (outward ray in x-y plane)
        let theta_i = 30.0_f64 * core::f64::consts::PI / 180.0;
        let dir = Vec3::new(libm::cos(theta_i), libm::sin(theta_i), 0.0);

        let n_from = 1.000293;
        let n_to = 1.000100;

        match refract_at_boundary(dir, boundary_pos, n_from, n_to) {
            RefractResult::Refracted(d) => {
                // sin(theta_i) = |dir x normal| (for unit vectors)
                let sin_i = dir.cross(normal).length();
                let sin_t = d.cross(normal).length();

                // Snell: n_from * sin_i = n_to * sin_t
                let lhs = n_from * sin_i;
                let rhs = n_to * sin_t;
                assert!(
                    (lhs - rhs).abs() < 1e-8,
                    "Snell's law violated: n1*sin_i={}, n2*sin_t={}",
                    lhs,
                    rhs
                );
            }
            RefractResult::TotalReflection(_) => {
                panic!("Should not get TIR at 30 deg with near-unity indices");
            }
        }
    }

    #[test]
    fn refract_denser_to_rarer_bends_away_from_normal() {
        // Going from denser to rarer medium, the ray bends away from the
        // normal (theta_t > theta_i).
        let r = 100.0;
        let boundary_pos = Vec3::new(r, 0.0, 0.0);
        let normal = boundary_pos.normalize();

        let theta_i = 20.0_f64 * core::f64::consts::PI / 180.0;
        // Outward ray
        let dir = Vec3::new(libm::cos(theta_i), libm::sin(theta_i), 0.0);

        let n_from = 1.0003; // denser
        let n_to = 1.0001; // rarer

        match refract_at_boundary(dir, boundary_pos, n_from, n_to) {
            RefractResult::Refracted(d) => {
                let sin_i = dir.cross(normal).length();
                let sin_t = d.cross(normal).length();
                assert!(
                    sin_t > sin_i,
                    "Denser to rarer: sin_t={} should be > sin_i={}",
                    sin_t,
                    sin_i
                );
            }
            RefractResult::TotalReflection(_) => {
                panic!("Should not get TIR at 20 deg");
            }
        }
    }

    #[test]
    fn refract_rarer_to_denser_bends_toward_normal() {
        // Going from rarer to denser medium, the ray bends toward the normal.
        let r = 100.0;
        let boundary_pos = Vec3::new(r, 0.0, 0.0);
        let normal = boundary_pos.normalize();

        let theta_i = 20.0_f64 * core::f64::consts::PI / 180.0;
        // Inward ray
        let dir = Vec3::new(-libm::cos(theta_i), libm::sin(theta_i), 0.0);

        let n_from = 1.0001; // rarer
        let n_to = 1.0003; // denser

        match refract_at_boundary(dir, boundary_pos, n_from, n_to) {
            RefractResult::Refracted(d) => {
                // The normal flips for inward rays, so compute angles
                // relative to the inward normal
                let inward_normal = -normal;
                let sin_i = dir.cross(inward_normal).length();
                let sin_t = d.cross(inward_normal).length();
                assert!(
                    sin_t < sin_i,
                    "Rarer to denser: sin_t={} should be < sin_i={}",
                    sin_t,
                    sin_i
                );
            }
            RefractResult::TotalReflection(_) => {
                panic!("Should not get TIR going rarer to denser");
            }
        }
    }

    #[test]
    fn refract_total_internal_reflection() {
        // TIR occurs when going from denser to much rarer medium at grazing angle.
        // Use exaggerated n values to force TIR.
        let boundary_pos = Vec3::new(100.0, 0.0, 0.0);

        // Nearly tangential outward ray
        let dir = Vec3::new(0.01, 0.99995, 0.0).normalize();

        let n_from = 1.5; // glass-like
        let n_to = 1.0; // vacuum

        match refract_at_boundary(dir, boundary_pos, n_from, n_to) {
            RefractResult::TotalReflection(r) => {
                // Reflected ray should be unit length
                assert!(
                    (r.length() - 1.0).abs() < 1e-10,
                    "Reflected ray should be unit: |r|={}",
                    r.length()
                );
                // The reflected ray should be on the same side of the boundary
                // (dot with outward normal should have same sign or be reflected)
                let normal = boundary_pos.normalize();
                let dot_in = dir.dot(normal);
                let dot_out = r.dot(normal);
                // For TIR, the ray reflects, so the radial component flips sign
                // relative to the original -- but both can be positive since the
                // ray was nearly tangential to begin with. What matters is the
                // angle of reflection equals the angle of incidence.
                let cos_i = libm::fabs(dot_in);
                let cos_r = libm::fabs(dot_out);
                assert!(
                    (cos_i - cos_r).abs() < 1e-8,
                    "Reflection angle should equal incidence: cos_i={}, cos_r={}",
                    cos_i,
                    cos_r
                );
            }
            RefractResult::Refracted(_) => {
                panic!("Expected TIR for grazing ray from n=1.5 to n=1.0");
            }
        }
    }

    #[test]
    fn refract_no_tir_at_moderate_angles() {
        // With real atmospheric indices between adjacent shells, TIR requires
        // extremely grazing rays. For typical shell-to-shell transitions
        // (e.g., n=1.000293 to n=1.000200), the critical angle is:
        // arcsin(1.000200/1.000293) ~ 89.22 deg.
        // Verify that a 70-degree zenith angle ray does NOT produce TIR.
        let boundary_pos = Vec3::new(6_371_000.0, 0.0, 0.0);
        let theta = 70.0_f64 * core::f64::consts::PI / 180.0;
        let dir = Vec3::new(libm::cos(theta), libm::sin(theta), 0.0);

        match refract_at_boundary(dir, boundary_pos, 1.000293, 1.000200) {
            RefractResult::Refracted(_) => { /* expected */ }
            RefractResult::TotalReflection(_) => {
                panic!("70 deg should not produce TIR with atmospheric shell indices");
            }
        }

        // Even at 85 degrees, no TIR for adjacent-shell index change
        let theta_85 = 85.0_f64 * core::f64::consts::PI / 180.0;
        let dir_85 = Vec3::new(libm::cos(theta_85), libm::sin(theta_85), 0.0);
        match refract_at_boundary(dir_85, boundary_pos, 1.000293, 1.000200) {
            RefractResult::Refracted(_) => { /* expected */ }
            RefractResult::TotalReflection(_) => {
                panic!("85 deg should not produce TIR with adjacent shell indices");
            }
        }
    }

    #[test]
    fn refract_tir_critical_angle_atmospheric() {
        // For the extreme case of surface-to-vacuum (n=1.000293 to n=1.0),
        // the critical angle is arcsin(1.0/1.000293) ~ 88.6 deg.
        // A ray at 89 degrees (beyond critical) should TIR.
        let boundary_pos = Vec3::new(6_371_000.0, 0.0, 0.0);
        let theta = 89.0_f64 * core::f64::consts::PI / 180.0;
        let dir = Vec3::new(libm::cos(theta), libm::sin(theta), 0.0);

        match refract_at_boundary(dir, boundary_pos, 1.000293, 1.0) {
            RefractResult::TotalReflection(_) => { /* expected: beyond critical angle */ }
            RefractResult::Refracted(_) => {
                panic!("89 deg from n=1.000293 to n=1.0 should produce TIR (critical ~88.6 deg)");
            }
        }

        // A ray at 80 degrees (below critical) should refract normally
        let theta_80 = 80.0_f64 * core::f64::consts::PI / 180.0;
        let dir_80 = Vec3::new(libm::cos(theta_80), libm::sin(theta_80), 0.0);
        match refract_at_boundary(dir_80, boundary_pos, 1.000293, 1.0) {
            RefractResult::Refracted(_) => { /* expected */ }
            RefractResult::TotalReflection(_) => {
                panic!("80 deg from n=1.000293 to n=1.0 should refract, not TIR");
            }
        }
    }

    #[test]
    fn refract_preserves_plane_of_incidence() {
        // The refracted ray must lie in the same plane as the incident ray
        // and the surface normal. This means (dir x normal) and (refracted x normal)
        // should be parallel.
        let boundary_pos = Vec3::new(100.0, 0.0, 0.0);
        let dir = Vec3::new(0.5, 0.7, 0.3).normalize();
        let normal = boundary_pos.normalize();

        match refract_at_boundary(dir, boundary_pos, 1.000293, 1.000100) {
            RefractResult::Refracted(d) => {
                let cross_in = dir.cross(normal);
                let cross_out = d.cross(normal);
                // Parallel means their cross product is zero
                let cross_cross = cross_in.cross(cross_out);
                assert!(
                    cross_cross.length() < 1e-8,
                    "Refracted ray should be in the plane of incidence: cross={:?}",
                    cross_cross
                );
            }
            RefractResult::TotalReflection(_) => {
                panic!("Should not get TIR");
            }
        }
    }

    #[test]
    fn refract_symmetry_inward_outward() {
        // A ray refracted outward through a boundary, then refracted inward
        // through the same boundary at the same point, should recover the
        // original direction (time-reversal symmetry of Snell's law).
        let boundary_pos = Vec3::new(100.0, 0.0, 0.0);
        let n_inner = 1.000293;
        let n_outer = 1.000100;

        let original_dir = Vec3::new(0.5, 0.7, 0.3).normalize();

        // Refract outward (inner -> outer)
        let refracted = match refract_at_boundary(original_dir, boundary_pos, n_inner, n_outer) {
            RefractResult::Refracted(d) => d,
            RefractResult::TotalReflection(_) => panic!("Should not TIR"),
        };

        // Reverse direction and refract inward (outer -> inner)
        let reversed = -refracted;
        let back = match refract_at_boundary(reversed, boundary_pos, n_outer, n_inner) {
            RefractResult::Refracted(d) => d,
            RefractResult::TotalReflection(_) => panic!("Should not TIR on return"),
        };

        // Should recover -original_dir
        let expected = -original_dir;
        assert!(
            (back.x - expected.x).abs() < 1e-8,
            "Time reversal failed: back={:?}, expected={:?}",
            back,
            expected
        );
        assert!((back.y - expected.y).abs() < 1e-8);
        assert!((back.z - expected.z).abs() < 1e-8);
    }

    #[test]
    fn refract_deflection_magnitude_realistic() {
        // For atmospheric refraction, the total bending over the full atmosphere
        // is ~0.57 degrees at the horizon. A single shell boundary with
        // delta_n = 0.000093 should produce a very small deflection.
        let r = 6_371_000.0 + 10_000.0; // 10 km altitude
        let boundary_pos = Vec3::new(r, 0.0, 0.0);

        // Ray at 45 degrees from radial
        let theta = 45.0_f64 * core::f64::consts::PI / 180.0;
        let dir = Vec3::new(libm::cos(theta), libm::sin(theta), 0.0);

        let n_from = 1.000293; // sea level
        let n_to = 1.000200; // ~3.2 km scale height

        match refract_at_boundary(dir, boundary_pos, n_from, n_to) {
            RefractResult::Refracted(d) => {
                // The angular deflection should be tiny (< 0.01 degrees per boundary)
                let cos_angle = dir.dot(d);
                let angle_rad = libm::acos(if cos_angle > 1.0 {
                    1.0
                } else if cos_angle < -1.0 {
                    -1.0
                } else {
                    cos_angle
                });
                let angle_deg = angle_rad * 180.0 / core::f64::consts::PI;
                assert!(
                    angle_deg < 0.01,
                    "Single boundary deflection should be < 0.01 deg, got {} deg",
                    angle_deg
                );
                assert!(
                    angle_deg > 0.0,
                    "Deflection should be nonzero for different n"
                );
            }
            RefractResult::TotalReflection(_) => panic!("Should not TIR"),
        }
    }
}
