use bevy_ecs::prelude::*;

pub struct App {
    pub world: World,
    pub schedule: Schedule,
}

impl App {
    pub fn new() -> Self {
        let world = World::new();
        let schedule = Schedule::default();
        Self { world, schedule }
    }
    pub fn run(&mut self) -> ! {
        loop {
            self.schedule.run(&mut self.world);
        }
    }
}
