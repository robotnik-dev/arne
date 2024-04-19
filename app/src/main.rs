use bevy_ecs::prelude::*;

mod app;

fn process() {
    todo!()
}

fn main() {
    let mut app = app::App::new();

    app.schedule.add_systems(process);

    app.run();
}