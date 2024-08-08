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

use crate::Result;
use std::{collections::HashMap, io::ErrorKind::NotFound};

pub trait Generate {
    fn generate(&self) -> String;
}

#[derive(Default, Debug, PartialEq)]
pub enum ComponentType {
    #[default]
    Resistor,
    Capacitor,
    VoltageSourceDc,
}

pub enum NodeType {
    In,
    Out,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Component {
    pub symbol: String,
    pub name: String,
    prefix: String,
    value: f64,
    in_nodes: Vec<Node>,
    out_nodes: Vec<Node>,
    initial_voltage: Option<f64>,
    dc_symbol: Option<String>,
}

impl Generate for Component {
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
            self.in_nodes[0],
            self.out_nodes[0],
            dc_symbol,
            self.value,
            self.prefix,
            initial_voltage
        )
    }
}

impl Component {
    pub fn add_node(&mut self, node: Node, node_type: NodeType) {
        match node_type {
            NodeType::In => self.in_nodes.push(node),
            NodeType::Out => self.out_nodes.push(node),
        }
    }
}

#[derive(Default)]
pub struct ComponentBuilder {
    name: String,
    component_type: ComponentType,
    prefix: Option<String>,
    value: Option<f64>,
    initial_voltage: Option<f64>,
}

impl ComponentBuilder {
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

    pub fn build(self) -> Component {
        match self.component_type {
            ComponentType::Resistor => Component {
                symbol: String::from("r"),
                name: self.name,
                prefix: self.prefix.unwrap_or_default(),
                value: self.value.unwrap_or_default(),
                in_nodes: Vec::new(),
                out_nodes: Vec::new(),
                initial_voltage: None,
                dc_symbol: None,
            },
            ComponentType::Capacitor => Component {
                symbol: String::from("c"),
                name: self.name,
                prefix: self.prefix.unwrap_or_default(),
                value: self.value.unwrap_or_default(),
                in_nodes: Vec::new(),
                out_nodes: Vec::new(),
                initial_voltage: self.initial_voltage,
                dc_symbol: None,
            },
            ComponentType::VoltageSourceDc => Component {
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

#[derive(Debug, PartialEq, Default, Clone)]
pub struct Netlist {
    pub components: HashMap<String, Component>,
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

    pub fn add_component(&mut self, component: Component, label: String) -> Result {
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
    #[allow(unused)]
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

    #[allow(unused)]
    pub fn get_component_with_label(&self, label: String) -> Option<&Component> {
        self.components.get(&label)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    fn resistor3_3k() -> Component {
        ComponentBuilder::new(ComponentType::Resistor, String::from("1"))
            .value(3.3, Some(String::from("k")))
            .build()
    }

    fn capacitor_ic2_5() -> Component {
        ComponentBuilder::new(ComponentType::Capacitor, String::from("1"))
            .initial_voltage(2.5)
            .build()
    }

    fn capacitor_noic() -> Component {
        ComponentBuilder::new(ComponentType::Capacitor, String::from("1")).build()
    }

    fn voltagedc_9() -> Component {
        ComponentBuilder::new(ComponentType::VoltageSourceDc, String::from("1"))
            .value(9., None)
            .build()
    }

    #[test]
    fn build_resistor() {
        let resistor = Component {
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
        let capacitor = Component {
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
        let voltage_source_dc = Component {
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
    fn generate_resistor() {
        let mut component = self::resistor3_3k();
        component.add_node(Node(1), NodeType::In);
        component.add_node(Node(2), NodeType::Out);
        assert_eq!(component.generate(), "r1 1 2 3.3k");
    }

    #[test]
    fn generate_capacitor_noic() {
        let mut component = self::capacitor_noic();
        component.add_node(Node(1), NodeType::In);
        component.add_node(Node(2), NodeType::Out);
        assert_eq!(component.generate(), "c1 1 2 0");
    }

    #[test]
    fn generate_capacitor_ic() {
        let mut component = self::capacitor_ic2_5();
        component.add_node(Node(1), NodeType::In);
        component.add_node(Node(2), NodeType::Out);
        assert_eq!(component.generate(), "c1 1 2 0 ic=2.5");
    }

    #[test]
    fn generate_voltage_sourcedc() {
        let mut component = self::voltagedc_9();
        component.add_node(Node(1), NodeType::In);
        component.add_node(Node(2), NodeType::Out);
        assert_eq!(component.generate(), "v1 1 2 dc 9");
    }

    #[test]
    fn generate_netlist_string() {
        let resistor = self::resistor3_3k();
        let capacitor = self::capacitor_ic2_5();
        let voltage_source_dc = self::voltagedc_9();
        let mut netlist = Netlist::new();

        // adding components to netlist
        netlist
            .add_component(resistor, String::from("resistor"))
            .unwrap();
        netlist
            .add_component(capacitor, String::from("capacitor"))
            .unwrap();
        netlist
            .add_component(voltage_source_dc, String::from("voltage_source"))
            .unwrap();

        // adding nodes to components
        netlist
            .add_node_to_component(Node(1), String::from("resistor"), NodeType::In)
            .unwrap();
        netlist
            .add_node_to_component(Node(2), String::from("resistor"), NodeType::Out)
            .unwrap();

        netlist
            .add_node_to_component(Node(2), String::from("capacitor"), NodeType::In)
            .unwrap();
        netlist
            .add_node_to_component(Node(0), String::from("capacitor"), NodeType::Out)
            .unwrap();

        netlist
            .add_node_to_component(Node(1), String::from("voltage_source"), NodeType::In)
            .unwrap();
        netlist
            .add_node_to_component(Node(0), String::from("voltage_source"), NodeType::Out)
            .unwrap();

        let expected_resistor = "r1 1 2 3.3k";
        let expected_capacitor = "c1 2 0 0 ic=2.5";
        let expected_coltage_source = "v1 1 0 dc 9";
        let generated = netlist.generate();
        let outcome = generated.split("\n").collect::<Vec<&str>>();

        assert!(outcome.contains(&expected_resistor));
        assert!(outcome.contains(&expected_capacitor));
        assert!(outcome.contains(&expected_coltage_source));

        // test boilerplate at the beginning and end of the netlist
        assert_eq!(outcome[0], ".SUBCKT main");
        assert_eq!(outcome[outcome.len() - 2], ".ENDS");
        assert_eq!(outcome[outcome.len() - 1], ".END");
    }
}
