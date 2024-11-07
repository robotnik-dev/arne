use bevy::prelude::Res;
use plotters::prelude::*;
use plotters::style::Color;

use crate::AdaptiveConfig;

pub fn netlists_over_time(
    data: &Vec<f32>,
    path: &str,
    title: &str,
    population_size: usize,
    max_generations: u64,
) {
    let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let x_max = max_generations as f32;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(0f32..x_max, 0f32..population_size as f32)
        .unwrap();

    chart
        .configure_mesh()
        .axis_desc_style(("sans-serif", 14).into_font())
        .y_desc("netlists")
        .x_desc("generation")
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            data.iter().enumerate().map(|(x, y)| (x as f32, *y)),
            &RED,
        ))
        .unwrap();

    // chart
    //     .configure_series_labels()
    //     .background_style(&WHITE.mix(0.8))
    //     .draw()
    //     .unwrap();

    root.present().unwrap();
}

pub fn plot_average_fitness(data: &[f32], path: &str, max_generations: u64) {
    let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let x_max = max_generations as f32;
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
    let moving_avarage = data.iter();

    chart
        .draw_series(LineSeries::new(
            moving_avarage.enumerate().map(|(x, y)| (x as f32, *y)),
            &RED,
        ))
        .unwrap();

    // chart
    //     .configure_series_labels()
    //     .background_style(WHITE.mix(0.8))
    //     .draw()
    //     .unwrap();

    root.present().unwrap();
}

pub fn mean_and_standard_deviation_over_time(
    mean_data: &Vec<f32>,
    sd_data: &Vec<f32>,
    path: &str,
    title: &str,
    adaptive_config: &Res<AdaptiveConfig>,
) {
    let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let x_max = adaptive_config.max_generations as f32;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0f32..x_max,
            adaptive_config.neuron_lower..adaptive_config.neuron_upper,
        )
        .unwrap();

    chart
        .configure_mesh()
        .axis_desc_style(("sans-serif", 14).into_font())
        .x_desc("generation")
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            mean_data.iter().enumerate().map(|(x, y)| (x as f32, *y)),
            &RED,
        ))
        .unwrap()
        .label("mean")
        .legend(|(x, y)| Rectangle::new([(x - 15, y + 1), (x, y)], RED));

    chart
        .draw_series(LineSeries::new(
            sd_data.iter().enumerate().map(|(x, y)| (x as f32, *y)),
            &BLUE,
        ))
        .unwrap()
        .label("standard deviation")
        .legend(|(x, y)| Rectangle::new([(x - 15, y + 1), (x, y)], BLUE));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .margin(20)
        .legend_area_size(5)
        .border_style(BLUE)
        .background_style(BLUE.mix(0.1))
        .label_font(("Calibri", 20))
        .draw()
        .unwrap();

    root.present().unwrap();
}
