use plotters::prelude::*;
use plotters::style::Color;

use crate::CONFIG;

pub fn update_image(data: &Vec<f32>, path: &str) {
    let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let x_max = CONFIG.genetic_algorithm.max_generations as f32;
    let mut chart = ChartBuilder::on(&root)
        .caption("Average fitness", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(0f32..x_max, 0f32..1f32)
        .unwrap();

    chart
        .configure_mesh()
        .axis_desc_style(("sans-serif", 14).into_font())
        .y_desc("fitness")
        .x_desc("generation")
        .draw()
        .unwrap();

    // convert data to moving avarage
    let moving_avarage = data.iter().map(|x| {
        // TODO: smooth curve
        x
    });

    chart
        .draw_series(LineSeries::new(
            moving_avarage.enumerate().map(|(x, y)| (x as f32, *y)),
            &RED,
        ))
        .unwrap();

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .draw()
        .unwrap();

    root.present().unwrap();
}
