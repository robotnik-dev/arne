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

use bevy::prelude::Component;

use crate::Result;
use std::{collections::HashMap, fmt::Display, io::ErrorKind::NotFound};

pub trait Build {
    fn build(&self, image_id: u64) -> Netlist;
}

pub trait Generate {
    fn generate(&self) -> String;
}

#[derive(Default, Debug, PartialEq, Clone, Eq, Hash)]
pub enum ComponentType {
    #[default]
    Resistor,
    Capacitor,
    VoltageSourceDc,
}

impl Display for ComponentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComponentType::Resistor => write!(f, "R"),
            ComponentType::Capacitor => write!(f, "C"),
            ComponentType::VoltageSourceDc => write!(f, "V"),
        }
    }
}

pub enum NodeType {
    #[allow(dead_code)]
    In,
    #[allow(dead_code)]
    Out,
}

#[derive(Debug, PartialEq, Clone)]
pub struct EComponent {
    pub symbol: String,
    pub name: String,
    prefix: String,
    value: f64,
    in_nodes: Vec<Node>,
    out_nodes: Vec<Node>,
    initial_voltage: Option<f64>,
    dc_symbol: Option<String>,
}

impl Generate for EComponent {
    fn generate(&self) -> String {
        let dc_symbol = if self.dc_symbol.is_some() {
            format!("{} ", self.dc_symbol.clone().unwrap())
        } else {
            String::from("")
        };
        let initial_voltage = if self.initial_voltage.is_some() {
            format!(" ic={}", self.initial_voltage.unwrap())
        } else {
            String::from("")
        };
        // just the first in and out node for simple circuits. Need to expand when doing transistor e.g.
        format!(
            "{}{} {} {} {}{}{}{}",
            self.symbol,
            self.name,
            "0", // TODO: netlist nodes
            "0", // TODO: netlist nodes
            dc_symbol,
            self.value,
            self.prefix,
            initial_voltage
        )
    }
}

impl EComponent {
    pub fn add_node(&mut self, node: Node, node_type: NodeType) {
        match node_type {
            NodeType::In => self.in_nodes.push(node),
            NodeType::Out => self.out_nodes.push(node),
        }
    }
}

#[derive(Default)]
pub struct ComponentBuilder {
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    component_type: ComponentType,
    prefix: Option<String>,
    value: Option<f64>,
    initial_voltage: Option<f64>,
}

impl ComponentBuilder {
    #[allow(dead_code)]
    pub fn new(component_type: ComponentType, name: String) -> ComponentBuilder {
        ComponentBuilder {
            name,
            component_type,
            ..Default::default()
        }
    }

    #[allow(unused)]
    pub fn value(mut self, value: f64, prefix: Option<String>) -> ComponentBuilder {
        self.value = Some(value);
        self.prefix = prefix;
        self
    }

    #[allow(unused)]
    pub fn initial_voltage(mut self, initial_voltage: f64) -> ComponentBuilder {
        self.initial_voltage = Some(initial_voltage);
        self
    }

    #[allow(dead_code)]
    pub fn build(self) -> EComponent {
        match self.component_type {
            ComponentType::Resistor => EComponent {
                symbol: String::from("r"),
                name: self.name,
                prefix: self.prefix.unwrap_or_default(),
                value: self.value.unwrap_or_default(),
                in_nodes: Vec::new(),
                out_nodes: Vec::new(),
                initial_voltage: None,
                dc_symbol: None,
            },
            ComponentType::Capacitor => EComponent {
                symbol: String::from("c"),
                name: self.name,
                prefix: self.prefix.unwrap_or_default(),
                value: self.value.unwrap_or_default(),
                in_nodes: Vec::new(),
                out_nodes: Vec::new(),
                initial_voltage: self.initial_voltage,
                dc_symbol: None,
            },
            ComponentType::VoltageSourceDc => EComponent {
                symbol: String::from("v"),
                name: self.name,
                prefix: self.prefix.unwrap_or_default(),
                value: self.value.unwrap_or_default(),
                in_nodes: Vec::new(),
                out_nodes: Vec::new(),
                initial_voltage: None,
                dc_symbol: Some(String::from("dc")),
            },
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Default, Clone)]
pub struct Node(pub u32);

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, PartialEq, Default, Clone, Component)]
pub struct Netlist {
    pub components: HashMap<String, EComponent>,
}

impl Generate for Netlist {
    fn generate(&self) -> String {
        let mut netlist = String::new();
        netlist.push_str(".SUBCKT main\n");
        self.components.iter().for_each(|(_, component)| {
            netlist.push_str(&component.generate());
            netlist.push('\n');
        });
        netlist.push_str(".ENDS\n");
        netlist.push_str(".END");
        netlist
    }
}

impl Netlist {
    pub fn new() -> Self {
        Netlist {
            components: HashMap::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Compares with an other netlist.
    /// Returns a percentage of similiarity between 0 and 1
    pub fn compare(&self, with: &Netlist) -> f32 {
        // save the maximum count of components that can be compared against (self ist the optimal netlist)
        let max_components = self.components.iter().len();

        let mut found_components = with
            .components
            .values()
            .cloned()
            .collect::<Vec<EComponent>>();
        let optimal_components = self
            .components
            .values()
            .cloned()
            .collect::<Vec<EComponent>>();
        let mut correct_component_count = 0usize;
        // go through all components that should be present and count the correct ones.
        for optimal_comp in optimal_components {
            // this component of the optimal ones is found in the other component list
            if let Some(index) = found_components.iter().position(|c| c == &optimal_comp) {
                // increase the count of correct components
                correct_component_count += 1;
                // delete the component to prevent duplicate counting
                found_components.remove(index);
            }
        }
        if max_components > 0 {
            // normal calculation of similiarity
            return correct_component_count as f32 / max_components as f32;
        } else if max_components == 0 {
            // return 1.0 only when the found components are also zero, else return 0.0
            if found_components.iter().len() == 0 {
                return 1.;
            } else {
                return 0.;
            }
        } else {
            return 0.;
        }
    }

    pub fn add_component(&mut self, component: EComponent, label: String) -> Result {
        if self.components.contains_key(&label) {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("Component with label {} already exists", label),
            )));
        };
        self.components.insert(label, component);
        Ok(())
    }

    /// adds a node to a component with the specified label
    #[allow(dead_code)]
    pub fn add_node_to_component(
        &mut self,
        node: Node,
        label: String,
        node_type: NodeType,
    ) -> Result {
        let Some(component) = self.components.get_mut(&label) else {
            return Err(Box::new(std::io::Error::new(
                NotFound,
                format!("Component with label {} not found", label),
            )));
        };
        component.add_node(node, node_type);
        Ok(())
    }

    #[allow(dead_code)]
    pub fn get_component_with_label(&self, label: String) -> Option<&EComponent> {
        self.components.get(&label)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[allow(dead_code)]
    fn resistor3_3k() -> EComponent {
        ComponentBuilder::new(ComponentType::Resistor, String::from("1"))
            .value(3.3, Some(String::from("k")))
            .build()
    }

    #[allow(dead_code)]
    fn capacitor_ic2_5() -> EComponent {
        ComponentBuilder::new(ComponentType::Capacitor, String::from("1"))
            .initial_voltage(2.5)
            .build()
    }

    #[allow(dead_code)]
    fn capacitor_noic() -> EComponent {
        ComponentBuilder::new(ComponentType::Capacitor, String::from("1")).build()
    }

    #[allow(dead_code)]
    fn voltagedc_9() -> EComponent {
        ComponentBuilder::new(ComponentType::VoltageSourceDc, String::from("1"))
            .value(9., None)
            .build()
    }

    #[test]
    fn build_resistor() {
        let resistor = EComponent {
            symbol: String::from("r"),
            name: String::from("1"),
            prefix: String::from("k"),
            value: 3.3,
            in_nodes: Vec::new(),
            out_nodes: Vec::new(),
            initial_voltage: None,
            dc_symbol: None,
        };
        let resistor_from_builder =
            ComponentBuilder::new(ComponentType::Resistor, String::from("1"))
                .value(3.3, Some(String::from("k")))
                .build();
        assert_eq!(resistor, resistor_from_builder);
    }

    #[test]
    fn build_capacitor() {
        let capacitor = EComponent {
            symbol: String::from("c"),
            name: String::from("1"),
            prefix: String::from(""),
            value: 0.0,
            in_nodes: Vec::new(),
            out_nodes: Vec::new(),
            initial_voltage: Some(3.5),
            dc_symbol: None,
        };
        let capacitor_from_builder =
            ComponentBuilder::new(ComponentType::Capacitor, String::from("1"))
                .initial_voltage(3.5)
                .build();
        assert_eq!(capacitor, capacitor_from_builder);
    }

    #[test]
    fn build_voltage_source_dc() {
        let voltage_source_dc = EComponent {
            symbol: String::from("v"),
            name: String::from("1"),
            prefix: String::from(""),
            value: 9.0,
            in_nodes: Vec::new(),
            out_nodes: Vec::new(),
            initial_voltage: None,
            dc_symbol: Some(String::from("dc")),
        };
        let voltage_source_dc_from_builder =
            ComponentBuilder::new(ComponentType::VoltageSourceDc, String::from("1"))
                .value(9.0, None)
                .build();
        assert_eq!(voltage_source_dc, voltage_source_dc_from_builder);
    }

    #[test]
    fn compare() {
        let r0 = ComponentBuilder::new(ComponentType::Resistor, "r0".into()).build();
        let r1 = ComponentBuilder::new(ComponentType::Resistor, "r1".into()).build();
        let r2 = ComponentBuilder::new(ComponentType::Resistor, "r2".into()).build();
        let c0 = ComponentBuilder::new(ComponentType::Capacitor, "c0".into()).build();
        let c1 = ComponentBuilder::new(ComponentType::Capacitor, "c1".into()).build();
        let c2 = ComponentBuilder::new(ComponentType::Capacitor, "c2".into()).build();

        let mut optimal = Netlist::new();
        optimal.add_component(r0.clone(), "r0".into()).unwrap();
        optimal.add_component(r2, "r2".into()).unwrap();
        optimal.add_component(c1, "c1".into()).unwrap();
        optimal.add_component(c2.clone(), "c2".into()).unwrap();

        let mut with = Netlist::new();
        with.add_component(r0.clone(), "r0".into()).unwrap();
        assert_eq!(optimal.compare(&with), 0.25);

        with.add_component(r1, "r1".into()).unwrap();
        assert_eq!(optimal.compare(&with), 0.25);

        with.add_component(c2.clone(), "c2".into()).unwrap();
        assert_eq!(optimal.compare(&with), 0.5);

        // testing when the optimal netlist has less components than with
        let mut optimal = Netlist::new();
        optimal.add_component(r0.clone(), "r0".into()).unwrap();
        assert_eq!(optimal.compare(&with), 1.0);

        optimal.add_component(c0.clone(), "c0".into()).unwrap();
        assert_eq!(optimal.compare(&with), 0.5);

        // testing when the optimal netlist has no components(optional: does not occur)
        let optimal = Netlist::new();
        assert_eq!(optimal.compare(&with), 0.0);

        let with = Netlist::new();
        assert_eq!(optimal.compare(&with), 1.0);
    }

    // #[test]
    // fn generate_resistor() {
    //     let mut component = self::resistor3_3k();
    //     component.add_node(Node(1), NodeType::In);
    //     component.add_node(Node(2), NodeType::Out);
    //     assert_eq!(component.generate(), "r1 1 2 3.3k");
    // }

    // #[test]
    // fn generate_capacitor_noic() {
    //     let mut component = self::capacitor_noic();
    //     component.add_node(Node(1), NodeType::In);
    //     component.add_node(Node(2), NodeType::Out);
    //     assert_eq!(component.generate(), "c1 1 2 0");
    // }

    // #[test]
    // fn generate_capacitor_ic() {
    //     let mut component = self::capacitor_ic2_5();
    //     component.add_node(Node(1), NodeType::In);
    //     component.add_node(Node(2), NodeType::Out);
    //     assert_eq!(component.generate(), "c1 1 2 0 ic=2.5");
    // }

    // #[test]
    // fn generate_voltage_sourcedc() {
    //     let mut component = self::voltagedc_9();
    //     component.add_node(Node(1), NodeType::In);
    //     component.add_node(Node(2), NodeType::Out);
    //     assert_eq!(component.generate(), "v1 1 2 dc 9");
    // }

    // #[test]
    // fn generate_netlist_string() {
    //     let resistor = self::resistor3_3k();
    //     let capacitor = self::capacitor_ic2_5();
    //     let voltage_source_dc = self::voltagedc_9();
    //     let mut netlist = Netlist::new();

    //     // adding components to netlist
    //     netlist
    //         .add_component(resistor, String::from("resistor"))
    //         .unwrap();
    //     netlist
    //         .add_component(capacitor, String::from("capacitor"))
    //         .unwrap();
    //     netlist
    //         .add_component(voltage_source_dc, String::from("voltage_source"))
    //         .unwrap();

    //     // adding nodes to components
    //     netlist
    //         .add_node_to_component(Node(1), String::from("resistor"), NodeType::In)
    //         .unwrap();
    //     netlist
    //         .add_node_to_component(Node(2), String::from("resistor"), NodeType::Out)
    //         .unwrap();

    //     netlist
    //         .add_node_to_component(Node(2), String::from("capacitor"), NodeType::In)
    //         .unwrap();
    //     netlist
    //         .add_node_to_component(Node(0), String::from("capacitor"), NodeType::Out)
    //         .unwrap();

    //     netlist
    //         .add_node_to_component(Node(1), String::from("voltage_source"), NodeType::In)
    //         .unwrap();
    //     netlist
    //         .add_node_to_component(Node(0), String::from("voltage_source"), NodeType::Out)
    //         .unwrap();

    //     let expected_resistor = "r1 1 2 3.3k";
    //     let expected_capacitor = "c1 2 0 0 ic=2.5";
    //     let expected_coltage_source = "v1 1 0 dc 9";
    //     let generated = netlist.generate();
    //     let outcome = generated.split("\n").collect::<Vec<&str>>();

    //     assert!(outcome.contains(&expected_resistor));
    //     assert!(outcome.contains(&expected_capacitor));
    //     assert!(outcome.contains(&expected_coltage_source));

    //     // test boilerplate at the beginning and end of the netlist
    //     assert_eq!(outcome[0], ".SUBCKT main");
    //     assert_eq!(outcome[outcome.len() - 2], ".ENDS");
    //     assert_eq!(outcome[outcome.len() - 1], ".END");
    // }
}
