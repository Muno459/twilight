//! Coordinate projection: WGS84 <-> UTM.
//!
//! Pure Rust implementation of the Transverse Mercator projection on the
//! WGS84 ellipsoid. Used to convert between lat/lon and UTM easting/northing
//! for national LiDAR datasets that use UTM coordinates (e.g., Denmark EPSG:25832).
//!
//! Based on Karney (2011) "Transverse Mercator with an accuracy of a few nanometers"
//! but using the simplified Redfearn series which is accurate to ~1mm for UTM.

/// WGS84 ellipsoid parameters.
const WGS84_A: f64 = 6_378_137.0; // semi-major axis (m)
const WGS84_F: f64 = 1.0 / 298.257_223_563; // flattening
const WGS84_E2: f64 = 2.0 * WGS84_F - WGS84_F * WGS84_F; // first eccentricity squared

/// UTM scale factor at central meridian.
const UTM_K0: f64 = 0.9996;
/// UTM false easting (m).
const UTM_FALSE_EASTING: f64 = 500_000.0;
/// UTM false northing for southern hemisphere (m).
const UTM_FALSE_NORTHING_SOUTH: f64 = 10_000_000.0;

/// Determine the UTM zone number for a given longitude.
pub fn utm_zone(lon: f64) -> u8 {
    let lon_norm = ((lon + 180.0).rem_euclid(360.0)) - 180.0;
    ((lon_norm + 180.0) / 6.0).floor() as u8 + 1
}

/// Central meridian for a UTM zone (degrees).
pub fn utm_central_meridian(zone: u8) -> f64 {
    (zone as f64 - 1.0) * 6.0 - 180.0 + 3.0
}

/// Convert WGS84 (lat, lon) in degrees to UTM (easting, northing) in meters.
/// Returns (easting, northing, zone, is_northern_hemisphere).
pub fn wgs84_to_utm(lat: f64, lon: f64) -> (f64, f64, u8, bool) {
    let zone = utm_zone(lon);
    let (e, n) = wgs84_to_utm_zone(lat, lon, zone);
    (e, n, zone, lat >= 0.0)
}

/// Convert WGS84 (lat, lon) to UTM in a specific zone.
/// Returns (easting, northing). Northing includes false northing for southern hemisphere.
pub fn wgs84_to_utm_zone(lat: f64, lon: f64, zone: u8) -> (f64, f64) {
    let lat_rad = lat.to_radians();
    let lon0 = utm_central_meridian(zone);
    let dlon = (lon - lon0).to_radians();

    let e2 = WGS84_E2;
    let ep2 = e2 / (1.0 - e2); // second eccentricity squared

    let sin_lat = lat_rad.sin();
    let cos_lat = lat_rad.cos();
    let tan_lat = lat_rad.tan();

    let n_val = WGS84_A / (1.0 - e2 * sin_lat * sin_lat).sqrt(); // radius of curvature
    let t = tan_lat;
    let t2 = t * t;
    let c = ep2 * cos_lat * cos_lat;
    let a_val = cos_lat * dlon;
    let a2 = a_val * a_val;

    // Meridional arc length (M)
    let m = meridional_arc(lat_rad);

    // Easting
    let easting = UTM_K0
        * n_val
        * (a_val
            + (1.0 - t2 + c) * a2 * a_val / 6.0
            + (5.0 - 18.0 * t2 + t2 * t2 + 72.0 * c - 58.0 * ep2) * a2 * a2 * a_val / 120.0)
        + UTM_FALSE_EASTING;

    // Northing
    let mut northing = UTM_K0
        * (m + n_val
            * tan_lat
            * (a2 / 2.0
                + (5.0 - t2 + 9.0 * c + 4.0 * c * c) * a2 * a2 / 24.0
                + (61.0 - 58.0 * t2 + t2 * t2 + 600.0 * c - 330.0 * ep2) * a2 * a2 * a2 / 720.0));

    if lat < 0.0 {
        northing += UTM_FALSE_NORTHING_SOUTH;
    }

    (easting, northing)
}

/// Convert UTM (easting, northing) to WGS84 (lat, lon) in degrees.
pub fn utm_to_wgs84(easting: f64, northing: f64, zone: u8, northern: bool) -> (f64, f64) {
    let lon0 = utm_central_meridian(zone);

    let x = easting - UTM_FALSE_EASTING;
    let y = if northern {
        northing
    } else {
        northing - UTM_FALSE_NORTHING_SOUTH
    };

    let e2 = WGS84_E2;
    let ep2 = e2 / (1.0 - e2);
    let e1 = (1.0 - (1.0 - e2).sqrt()) / (1.0 + (1.0 - e2).sqrt());

    // Footpoint latitude
    let m = y / UTM_K0;
    let mu = m / (WGS84_A * (1.0 - e2 / 4.0 - 3.0 * e2 * e2 / 64.0 - 5.0 * e2 * e2 * e2 / 256.0));

    let fp_lat = mu
        + (3.0 * e1 / 2.0 - 27.0 * e1 * e1 * e1 / 32.0) * (2.0 * mu).sin()
        + (21.0 * e1 * e1 / 16.0 - 55.0 * e1 * e1 * e1 * e1 / 32.0) * (4.0 * mu).sin()
        + (151.0 * e1 * e1 * e1 / 96.0) * (6.0 * mu).sin()
        + (1097.0 * e1 * e1 * e1 * e1 / 512.0) * (8.0 * mu).sin();

    let sin_fp = fp_lat.sin();
    let cos_fp = fp_lat.cos();
    let tan_fp = fp_lat.tan();
    let n1 = WGS84_A / (1.0 - e2 * sin_fp * sin_fp).sqrt();
    let r1 = WGS84_A * (1.0 - e2) / (1.0 - e2 * sin_fp * sin_fp).powf(1.5);
    let t1 = tan_fp;
    let t12 = t1 * t1;
    let c1 = ep2 * cos_fp * cos_fp;
    let d = x / (n1 * UTM_K0);
    let d2 = d * d;

    let lat = fp_lat
        - (n1 * tan_fp / r1)
            * (d2 / 2.0
                - (5.0 + 3.0 * t12 + 10.0 * c1 - 4.0 * c1 * c1 - 9.0 * ep2) * d2 * d2 / 24.0
                + (61.0 + 90.0 * t12 + 298.0 * c1 + 45.0 * t12 * t12
                    - 252.0 * ep2
                    - 3.0 * c1 * c1)
                    * d2
                    * d2
                    * d2
                    / 720.0);

    let lon = (d - (1.0 + 2.0 * t12 + c1) * d2 * d / 6.0
        + (5.0 - 2.0 * c1 + 28.0 * t12 - 3.0 * c1 * c1 + 8.0 * ep2 + 24.0 * t12 * t12)
            * d2
            * d2
            * d
            / 120.0)
        / cos_fp;

    (lat.to_degrees(), lon0 + lon.to_degrees())
}

/// Meridional arc length from equator to latitude phi (radians).
fn meridional_arc(phi: f64) -> f64 {
    let e2 = WGS84_E2;
    let e4 = e2 * e2;
    let e6 = e4 * e2;

    let a0 = 1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0;
    let a2 = 3.0 / 8.0 * (e2 + e4 / 4.0 + 15.0 * e6 / 128.0);
    let a4 = 15.0 / 256.0 * (e4 + 3.0 * e6 / 4.0);
    let a6 = 35.0 * e6 / 3072.0;

    WGS84_A * (a0 * phi - a2 * (2.0 * phi).sin() + a4 * (4.0 * phi).sin() - a6 * (6.0 * phi).sin())
}

/// Vincenty forward problem: given a start point, azimuth, and distance,
/// compute the destination point on the WGS84 ellipsoid.
///
/// Returns (lat2, lon2) in degrees.
///
/// Uses the iterative Vincenty direct formula. Accurate to ~0.5mm.
pub fn vincenty_forward(lat1_deg: f64, lon1_deg: f64, azimuth_deg: f64, dist_m: f64) -> (f64, f64) {
    if dist_m < 0.001 {
        return (lat1_deg, lon1_deg);
    }

    let a = WGS84_A;
    let f = WGS84_F;
    let b = a * (1.0 - f);

    let phi1 = lat1_deg.to_radians();
    let alpha1 = azimuth_deg.to_radians();
    let s = dist_m;

    let tan_u1 = (1.0 - f) * phi1.tan();
    let cos_u1 = 1.0 / (1.0 + tan_u1 * tan_u1).sqrt();
    let sin_u1 = tan_u1 * cos_u1;

    let sin_alpha1 = alpha1.sin();
    let cos_alpha1 = alpha1.cos();

    let sigma1 = tan_u1.atan2(cos_alpha1);
    let sin_alpha = cos_u1 * sin_alpha1;
    let cos2_alpha = 1.0 - sin_alpha * sin_alpha;

    let u_sq = cos2_alpha * (a * a - b * b) / (b * b);
    let a_coeff = 1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
    let b_coeff = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));

    let mut sigma = s / (b * a_coeff);

    for _ in 0..100 {
        let cos_2sigma_m = (2.0 * sigma1 + sigma).cos();
        let sin_sigma = sigma.sin();
        let cos_sigma = sigma.cos();
        let cos2_2sigma_m = cos_2sigma_m * cos_2sigma_m;

        let delta_sigma = b_coeff
            * sin_sigma
            * (cos_2sigma_m
                + b_coeff / 4.0
                    * (cos_sigma * (-1.0 + 2.0 * cos2_2sigma_m)
                        - b_coeff / 6.0
                            * cos_2sigma_m
                            * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                            * (-3.0 + 4.0 * cos2_2sigma_m)));

        let sigma_new = s / (b * a_coeff) + delta_sigma;
        if (sigma_new - sigma).abs() < 1e-12 {
            break;
        }
        sigma = sigma_new;
    }

    let sin_sigma = sigma.sin();
    let cos_sigma = sigma.cos();
    let cos_2sigma_m = (2.0 * sigma1 + sigma).cos();

    let lat2 = (sin_u1 * cos_sigma + cos_u1 * sin_sigma * cos_alpha1).atan2(
        (1.0 - f)
            * (sin_alpha * sin_alpha
                + (sin_u1 * sin_sigma - cos_u1 * cos_sigma * cos_alpha1).powi(2))
            .sqrt(),
    );

    let lambda =
        (sin_sigma * sin_alpha1).atan2(cos_u1 * cos_sigma - sin_u1 * sin_sigma * cos_alpha1);

    let c = f / 16.0 * cos2_alpha * (4.0 + f * (4.0 - 3.0 * cos2_alpha));
    let l = lambda
        - (1.0 - c)
            * f
            * sin_alpha
            * (sigma
                + c * sin_sigma
                    * (cos_2sigma_m + c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)));

    let lat2_deg = lat2.to_degrees();
    let lon2_deg = lon1_deg + l.to_degrees();

    (lat2_deg, lon2_deg)
}

/// Vincenty inverse problem: compute distance and azimuth between two points.
/// Returns (distance_m, azimuth1_deg, azimuth2_deg).
pub fn vincenty_inverse(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> (f64, f64, f64) {
    let a = WGS84_A;
    let f = WGS84_F;
    let b = a * (1.0 - f);

    let phi1 = lat1.to_radians();
    let phi2 = lat2.to_radians();
    let l = (lon2 - lon1).to_radians();

    let u1 = ((1.0 - f) * phi1.tan()).atan();
    let u2 = ((1.0 - f) * phi2.tan()).atan();
    let sin_u1 = u1.sin();
    let cos_u1 = u1.cos();
    let sin_u2 = u2.sin();
    let cos_u2 = u2.cos();

    let mut lambda = l;
    let mut sin_sigma;
    let mut cos_sigma;
    let mut sigma;
    let mut sin_alpha;
    let mut cos2_alpha;
    let mut cos_2sigma_m;

    for _ in 0..100 {
        let sin_lambda = lambda.sin();
        let cos_lambda = lambda.cos();

        sin_sigma = ((cos_u2 * sin_lambda).powi(2)
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda).powi(2))
        .sqrt();

        if sin_sigma < 1e-15 {
            return (0.0, 0.0, 0.0); // coincident points
        }

        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
        sigma = sin_sigma.atan2(cos_sigma);
        sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma;
        cos2_alpha = 1.0 - sin_alpha * sin_alpha;

        cos_2sigma_m = if cos2_alpha.abs() < 1e-15 {
            0.0
        } else {
            cos_sigma - 2.0 * sin_u1 * sin_u2 / cos2_alpha
        };

        let c = f / 16.0 * cos2_alpha * (4.0 + f * (4.0 - 3.0 * cos2_alpha));
        let lambda_prev = lambda;
        lambda = l
            + (1.0 - c)
                * f
                * sin_alpha
                * (sigma
                    + c * sin_sigma
                        * (cos_2sigma_m
                            + c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)));

        if (lambda - lambda_prev).abs() < 1e-12 {
            // Converged
            let u_sq = cos2_alpha * (a * a - b * b) / (b * b);
            let a_coeff =
                1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
            let b_coeff = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));

            let delta_sigma = b_coeff
                * sin_sigma
                * (cos_2sigma_m
                    + b_coeff / 4.0
                        * (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)
                            - b_coeff / 6.0
                                * cos_2sigma_m
                                * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                                * (-3.0 + 4.0 * cos_2sigma_m * cos_2sigma_m)));

            let dist = b * a_coeff * (sigma - delta_sigma);

            let sin_lambda = lambda.sin();
            let cos_lambda = lambda.cos();

            let az1 = (cos_u2 * sin_lambda)
                .atan2(cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda)
                .to_degrees();
            let az2 = (cos_u1 * sin_lambda)
                .atan2(-sin_u1 * cos_u2 + cos_u1 * sin_u2 * cos_lambda)
                .to_degrees();

            return (dist, (az1 + 360.0) % 360.0, (az2 + 360.0) % 360.0);
        }
    }

    // Didn't converge (near-antipodal points). Use spherical approximation.
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a_h = (dlat / 2.0).sin().powi(2) + phi1.cos() * phi2.cos() * (dlon / 2.0).sin().powi(2);
    let c_h = 2.0 * a_h.sqrt().atan2((1.0 - a_h).sqrt());
    let dist = WGS84_A * c_h;
    let az = dlon
        .sin()
        .atan2(phi1.cos() * phi2.tan() - phi1.sin() * dlon.cos());
    (dist, (az.to_degrees() + 360.0) % 360.0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utm_zone_basic() {
        assert_eq!(utm_zone(0.0), 31); // Greenwich
        assert_eq!(utm_zone(9.35), 32); // Padborg, Denmark
        assert_eq!(utm_zone(39.8), 37); // Mecca
        assert_eq!(utm_zone(-73.9), 18); // NYC
        assert_eq!(utm_zone(174.0), 60); // New Zealand
        assert_eq!(utm_zone(-179.5), 1); // Date line
    }

    #[test]
    fn utm_central_meridian_basic() {
        assert!((utm_central_meridian(32) - 9.0).abs() < 1e-10);
        assert!((utm_central_meridian(31) - 3.0).abs() < 1e-10);
        assert!((utm_central_meridian(1) - (-177.0)).abs() < 1e-10);
    }

    #[test]
    fn wgs84_utm_roundtrip_padborg() {
        // Padborg: ~55.05°N, 9.35°E -> UTM zone 32N
        let (e, n, zone, north) = wgs84_to_utm(55.05, 9.35);
        assert_eq!(zone, 32);
        assert!(north);
        assert!(e > 400_000.0 && e < 600_000.0);
        assert!(n > 6_000_000.0 && n < 6_200_000.0);

        // Round-trip
        let (lat, lon) = utm_to_wgs84(e, n, zone, north);
        assert!(
            (lat - 55.05).abs() < 1e-6,
            "lat roundtrip: {} vs 55.05",
            lat
        );
        assert!((lon - 9.35).abs() < 1e-6, "lon roundtrip: {} vs 9.35", lon);
    }

    #[test]
    fn wgs84_utm_roundtrip_mecca() {
        let (e, n, zone, north) = wgs84_to_utm(21.4225, 39.8262);
        assert_eq!(zone, 37);
        assert!(north);

        let (lat, lon) = utm_to_wgs84(e, n, zone, north);
        assert!((lat - 21.4225).abs() < 1e-6);
        assert!((lon - 39.8262).abs() < 1e-6);
    }

    #[test]
    fn wgs84_utm_roundtrip_southern() {
        // Sydney: ~33.87°S, 151.21°E
        let (e, n, zone, north) = wgs84_to_utm(-33.87, 151.21);
        assert!(!north);

        let (lat, lon) = utm_to_wgs84(e, n, zone, north);
        assert!((lat - (-33.87)).abs() < 1e-6);
        assert!((lon - 151.21).abs() < 1e-6);
    }

    #[test]
    fn vincenty_forward_short() {
        // Move 1000m north from equator
        let (lat, lon) = vincenty_forward(0.0, 0.0, 0.0, 1000.0);
        // 1000m ≈ 0.009° at equator
        assert!((lat - 0.009044).abs() < 0.0001);
        assert!(lon.abs() < 1e-10);
    }

    #[test]
    fn vincenty_forward_east() {
        // Move 1000m east from equator
        let (lat, lon) = vincenty_forward(0.0, 0.0, 90.0, 1000.0);
        assert!(lat.abs() < 1e-6);
        assert!((lon - 0.008983).abs() < 0.0001);
    }

    #[test]
    fn vincenty_forward_at_latitude() {
        // Move 10km east from Padborg
        let (lat, lon) = vincenty_forward(55.05, 9.35, 90.0, 10_000.0);
        // At 55°N, 1 degree lon ≈ 63.7 km, so 10km ≈ 0.157°
        assert!((lat - 55.05).abs() < 0.001);
        assert!((lon - 9.35 - 0.157).abs() < 0.01);
    }

    #[test]
    fn vincenty_inverse_known() {
        let (dist, az1, _az2) = vincenty_inverse(0.0, 0.0, 0.0, 1.0);
        // 1 degree of longitude at equator ≈ 111,319 m
        assert!((dist - 111_319.5).abs() < 10.0);
        assert!((az1 - 90.0).abs() < 0.01);
    }

    #[test]
    fn vincenty_forward_inverse_roundtrip() {
        let (lat2, lon2) = vincenty_forward(55.05, 9.35, 135.0, 50_000.0);
        let (dist, az, _) = vincenty_inverse(55.05, 9.35, lat2, lon2);
        assert!((dist - 50_000.0).abs() < 0.1);
        assert!((az - 135.0).abs() < 0.01);
    }

    #[test]
    fn vincenty_forward_zero_distance() {
        let (lat, lon) = vincenty_forward(55.05, 9.35, 42.0, 0.0);
        assert!((lat - 55.05).abs() < 1e-10);
        assert!((lon - 9.35).abs() < 1e-10);
    }
}
