
pub fn round_to_decimal_places(value: f64, decimal_places: u32) -> f64 {
    let multiplier = 10_f64.powi(decimal_places as i32);
    (value * multiplier).round() / multiplier
}

pub fn round2(value: f64) -> f64 {
    round_to_decimal_places(value, 2)
}