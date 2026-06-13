//! Documents the shadow-ray semantics for chords whose STRAIGHT-LINE
//! perigee dips below the surface - the reference for the GPU's O(1)
//! umbra pre-cull. With piecewise-constant Snell refraction (Bouguer:
//! n r sin(theta) = const), a descending ray's perigee is LOWER than the
//! straight chord's (by <= R*(n0-1) ~ 1.9 km), so any chord already
//! geometrically below the surface is blocked on the CPU too.
use twilight_core::geometry::Vec3;
use twilight_core::single_scatter::{shadow_ray_transmittance, CloudTransmittance};

#[test]
fn straight_chord_below_surface_is_blocked_on_cpu() {
    let atm = twilight_data::builder::build_clear_sky(
        twilight_data::atmosphere_profiles::AtmosphereType::UsStandard,
        0.15,
    );
    let re = twilight_core::atmosphere::EARTH_RADIUS_M;
    // 80 km is EXACTLY a shell boundary - the historical leak trigger.
    let p = Vec3::new(re + 80_000.0, 0.0, 0.0);
    for dip_km in [-30.0_f64, -10.0, -2.0, 0.0, 2.0, 5.0, 10.0, 30.0] {
        let b = re + dip_km * 1000.0;
        let sina = (b / p.length()).min(1.0);
        let cosa = -(1.0 - sina * sina).sqrt();
        let d = Vec3::new(cosa, sina, 0.0);
        let t = shadow_ray_transmittance(&atm, p, d, 0, None, CloudTransmittance::Diffuse);
        eprintln!("straight perigee {dip_km:+6.1} km -> CPU T = {t:.3e}");
        if dip_km < -0.5 {
            assert!(
                t < 1e-12,
                "chord {dip_km} km below surface must be blocked, got {t:.3e}"
            );
        }
        if dip_km > 5.0 {
            assert!(t > 0.0, "chord {dip_km} km above surface should transmit");
        }
    }
}
