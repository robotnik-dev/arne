// This file will contain the netlist data structure.
// I will restrict the possiblitites of Components to either a resistor, a voltage source or a capacitor (and ground).

// I want to save netlists in the following format:
// - the first line will be always: .SUBCKT main
// - the last two lines will be always:
// .ENDS
// .END
// In betwenn is the actual netlist.
// A line in the netlist will be a list of strings.

// RESISTORS
// General form: r[name] [node1] [node2] [value] Example: rload 23 15 3.3k 

// CAPACITORS
// General form: c[name] [node1] [node2] [value] ic=[initial voltage] Example 1: c1 12 33 10u Example 2: c1 12 33 10u ic=3.5

// VOLTAGE SOURCES (DC)
// General form: v[name] [+node] [-node] dc [voltage] Example 1: v1 1 0 dc 12 

// VOLTAGE SOURCES (SINE)
// General form: v[name] [+node] [-node] sin([offset] [voltage] + [freq] [delay] [damping factor]) Example 1: v1 1 0 sin(0 12 60 0 0)

#[derive(Default, Debug, PartialEq)]
enum ComponentType {
    #[default]
    Resistor,
    Capacitor,
    VoltageSourceDc,
    VoltageSourceSine,
}

#[derive(Debug, PartialEq)]
struct Component {
    symbol: String,
    name: String,
    prefix: String,
    value: f64,
    voltage_source: Option<VoltageSourceType>,
    offset: f64,
    freq: f64,
    delay: f64,
    damping_factor: f64,
}

#[derive(Default)]
struct ComponentBuilder {
    name: String,
    component_type: ComponentType,
    prefix: Option<String>,
    value: Option<f64>,
    voltage_source: Option<VoltageSourceType>,
    offset: Option<f64>,
    freq: Option<f64>,
    delay: Option<f64>,
    damping_factor: Option<f64>,
}

impl ComponentBuilder {
    pub fn new(name: String, component_type: ComponentType) -> ComponentBuilder {
        ComponentBuilder {
            name,
            component_type,
            ..Default::default()
        }
    }

    pub fn value(mut self, value: f64, prefix: Option<String>) -> ComponentBuilder {
        self.value = Some(value);
        self.prefix = prefix;
        self
    }

    pub fn offset(mut self, offset: f64) -> ComponentBuilder {
        self.offset = Some(offset);
        self
    }

    pub fn freq(mut self, freq: f64) -> ComponentBuilder {
        self.freq = Some(freq);
        self
    }

    pub fn delay(mut self, delay: f64) -> ComponentBuilder {
        self.delay = Some(delay);
        self
    }

    pub fn damping_factor(mut self, damping_factor: f64) -> ComponentBuilder {
        self.damping_factor = Some(damping_factor);
        self
    }

    pub fn build(self) -> Component {
        match self.component_type {
            ComponentType::Resistor => Component {
                symbol: String::from("r"),
                name: self.name,
                prefix: self.prefix.unwrap_or_default(),
                value: self.value.unwrap_or_default(),
                voltage_source: None,
                offset: self.offset.unwrap_or_default(),
                freq: self.freq.unwrap_or_default(),
                delay: self.delay.unwrap_or_default(),
                damping_factor: self.damping_factor.unwrap_or_default(),
            },
            ComponentType::Capacitor => Component {
                symbol: String::from("c"),
                name: self.name,
                prefix: self.prefix.unwrap_or_default(),
                value: self.value.unwrap_or_default(),
                voltage_source: None,
                offset: self.offset.unwrap_or_default(),
                freq: self.freq.unwrap_or_default(),
                delay: self.delay.unwrap_or_default(),
                damping_factor: self.damping_factor.unwrap_or_default(),
            },
            ComponentType::VoltageSourceDc => Component {
                symbol: String::from("v"),
                name: self.name,
                prefix: self.prefix.unwrap_or_default(),
                value: self.value.unwrap_or_default(),
                voltage_source: Some(VoltageSourceType::Dc),
                offset: self.offset.unwrap_or_default(),
                freq: self.freq.unwrap_or_default(),
                delay: self.delay.unwrap_or_default(),
                damping_factor: self.damping_factor.unwrap_or_default(),
            },
            ComponentType::VoltageSourceSine => Component {
                symbol: String::from("v"),
                name: self.name,
                prefix: self.prefix.unwrap_or_default(),
                value: self.value.unwrap_or_default(),
                voltage_source: Some(VoltageSourceType::Sine),
                offset: self.offset.unwrap_or_default(),
                freq: self.freq.unwrap_or_default(),
                delay: self.delay.unwrap_or_default(),
                damping_factor: self.damping_factor.unwrap_or_default(),
            },
        }
    }
}

#[derive(Debug, PartialEq)]
enum VoltageSourceType {
    Dc,
    Sine,
}

#[derive(Debug, PartialEq)]
struct Node(u32);

#[derive(Debug, PartialEq)]
struct Netlist {
    components: Vec<Component>,
    nodes: Vec<Node>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_resistor() {
        let resistor = Component {
            symbol: String::from("r"),
            name: String::from("1"),
            prefix: String::from("k"),
            value: 3.3,
            voltage_source: None,
            offset: 0.0,
            freq: 0.0,
            delay: 0.0,
            damping_factor: 0.0,
        };
        let resistor_from_builder = ComponentBuilder::new(String::from("1"), ComponentType::Resistor)
            .value(3.3, Some(String::from("k")))
            .build();
        assert_eq!(resistor, resistor_from_builder);
    }

    #[test]
    fn build_capacitor() {
        let capacitor = Component {
            symbol: String::from("c"),
            name: String::from("1"),
            prefix: String::from(""),
            value: 0.0,
            voltage_source: None,
            offset: 0.0,
            freq: 0.0,
            delay: 0.0,
            damping_factor: 0.0,
        };
        let capacitor_from_builder = ComponentBuilder::new(String::from("1"), ComponentType::Capacitor)
            .build();
        assert_eq!(capacitor, capacitor_from_builder);
    }
    
    #[test]
    fn build_voltage_source_dc() {
        let voltage_source_dc = Component {
            symbol: String::from("v"),
            name: String::from("1"),
            prefix: String::from(""),
            value: 9.0,
            voltage_source: Some(VoltageSourceType::Dc),
            offset: 0.0,
            freq: 0.0,
            delay: 0.0,
            damping_factor: 0.0,
        };
        let voltage_source_dc_from_builder = ComponentBuilder::new(String::from("1"), ComponentType::VoltageSourceDc)
            .value(9.0, None)
            .build();
        assert_eq!(voltage_source_dc, voltage_source_dc_from_builder);
    }
    
    #[test]
    fn build_voltage_source_sine() {
        let voltage_source_sine = Component {
            symbol: String::from("v"),
            name: String::from("1"),
            prefix: String::from(""),
            value: 12.0,
            voltage_source: Some(VoltageSourceType::Sine),
            offset: 0.4,
            freq: 50.0,
            delay: 1.0,
            damping_factor: 0.2,
        };
        let voltage_source_sine_from_builder = ComponentBuilder::new(String::from("1"), ComponentType::VoltageSourceSine)
            .value(12.0, None)
            .offset(0.4)
            .freq(50.0)
            .delay(1.0)
            .damping_factor(0.2)
            .build();
        assert_eq!(voltage_source_sine, voltage_source_sine_from_builder);
    }
}