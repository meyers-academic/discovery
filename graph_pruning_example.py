"""
Graph Pruning Example: Eliminate unnecessary intermediate computations.

In a complex likelihood with many components, we often build intermediate
structures that aren't actually used. Graph pruning removes these.
"""

import jax
import jax.numpy as jnp
from typing import Set, List


# ============================================================================
# Simple Dependency Tracking
# ============================================================================


class GraphNode:
    """Base node with dependency tracking."""

    def __init__(self, name: str):
        self.name = name
        self.dependencies: List[GraphNode] = []
        self._needed = False

    def add_dependency(self, node: "GraphNode"):
        """Add a dependency."""
        self.dependencies.append(node)

    def mark_needed(self):
        """Mark this node and all dependencies as needed."""
        if self._needed:
            return
        self._needed = True
        for dep in self.dependencies:
            dep.mark_needed()

    def __repr__(self):
        status = "✓" if self._needed else "✗"
        return f"{status} {self.name}"


class ComputationGraph:
    """Tracks all computation nodes and can prune unused ones."""

    def __init__(self):
        self.nodes: List[GraphNode] = []
        self.outputs: List[GraphNode] = []

    def add_node(self, node: GraphNode):
        """Add a node to the graph."""
        self.nodes.append(node)
        return node

    def mark_output(self, node: GraphNode):
        """Mark a node as an output (must be computed)."""
        self.outputs.append(node)
        node.mark_needed()

    def prune(self):
        """Remove nodes that don't contribute to any output."""
        # Mark all nodes reachable from outputs
        for output in self.outputs:
            output.mark_needed()

        # Keep only needed nodes
        before_count = len(self.nodes)
        self.nodes = [n for n in self.nodes if n._needed]
        after_count = len(self.nodes)

        return before_count - after_count

    def print_status(self):
        """Print which nodes are needed/unused."""
        print("\nGraph Status:")
        print("-" * 50)
        for node in self.nodes:
            print(f"  {node}")
        print(f"\nTotal nodes: {len(self.nodes)}")
        print(f"Needed: {sum(1 for n in self.nodes if n._needed)}")
        print(f"Unused: {sum(1 for n in self.nodes if not n._needed)}")


# ============================================================================
# Example: Multi-Component Model
# ============================================================================


def example_pruning():
    """
    Example showing graph pruning in a multi-component model.

    Scenario: We have white noise + red noise + DM noise for multiple pulsars.
    But for a specific analysis, we only use some components for some pulsars.
    """
    print("=" * 70)
    print("Example: Graph Pruning in Multi-Component Model")
    print("=" * 70)

    graph = ComputationGraph()

    # Build nodes for 3 pulsars, each with 3 noise components
    n_pulsars = 3
    components = ["white", "red", "dm"]

    # Create data nodes
    data_nodes = {}
    for i in range(n_pulsars):
        for comp in components:
            name = f"P{i}_{comp}_data"
            node = GraphNode(name)
            graph.add_node(node)
            data_nodes[(i, comp)] = node

    # Create Woodbury structure nodes for each component
    woodbury_nodes = {}
    for i in range(n_pulsars):
        for comp in components:
            name = f"P{i}_{comp}_woodbury"
            node = GraphNode(name)
            node.add_dependency(data_nodes[(i, comp)])
            graph.add_node(node)
            woodbury_nodes[(i, comp)] = node

    # Create likelihood nodes for each pulsar
    likelihood_nodes = {}
    for i in range(n_pulsars):
        name = f"P{i}_likelihood"
        node = GraphNode(name)
        # Add dependencies on all components for this pulsar
        for comp in components:
            node.add_dependency(woodbury_nodes[(i, comp)])
        graph.add_node(node)
        likelihood_nodes[i] = node

    # Total likelihood combines all pulsars
    total_likelihood = GraphNode("total_likelihood")
    for i in range(n_pulsars):
        total_likelihood.add_dependency(likelihood_nodes[i])
    graph.add_node(total_likelihood)

    print("\nScenario 1: Use all pulsars and all components")
    print("-" * 50)
    # Mark total likelihood as output
    graph.mark_output(total_likelihood)
    graph.print_status()
    print("All nodes needed ✓")

    # Reset
    for node in graph.nodes:
        node._needed = False

    print("\n\nScenario 2: Only use white + red noise for pulsar 0")
    print("-" * 50)
    # Only mark specific components as outputs
    graph.mark_output(woodbury_nodes[(0, "white")])
    graph.mark_output(woodbury_nodes[(0, "red")])

    graph.print_status()
    print(f"\nPruned {sum(1 for n in graph.nodes if not n._needed)} unused nodes!")
    print("DM noise and other pulsars not needed ✓")

    # Reset
    for node in graph.nodes:
        node._needed = False

    print("\n\nScenario 3: Only white noise for all pulsars")
    print("-" * 50)
    for i in range(n_pulsars):
        graph.mark_output(woodbury_nodes[(i, "white")])

    graph.print_status()
    print(f"\nPruned {sum(1 for n in graph.nodes if not n._needed)} unused nodes!")
    print("Red and DM noise not needed ✓")


# ============================================================================
# Example: Constant Folding
# ============================================================================


def example_constant_folding():
    """
    Example showing constant folding optimization.

    If a subgraph is entirely constant, we can pre-evaluate it.
    """
    print("\n\n" + "=" * 70)
    print("Example: Constant Folding Optimization")
    print("=" * 70)

    class ConstantNode(GraphNode):
        """Node with constant value."""

        def __init__(self, name: str, value):
            super().__init__(name)
            self.value = value
            self.is_constant = True

        def eval(self):
            return self.value

    class VariableNode(GraphNode):
        """Node with variable value."""

        def __init__(self, name: str):
            super().__init__(name)
            self.is_constant = False

        def eval(self, params):
            return params[self.name]

    class ComputeNode(GraphNode):
        """Node that computes from dependencies."""

        def __init__(self, name: str, deps: List[GraphNode]):
            super().__init__(name)
            for dep in deps:
                self.add_dependency(dep)

        @property
        def is_constant(self):
            """Constant if all dependencies are constant."""
            return all(getattr(dep, "is_constant", False) for dep in self.dependencies)

    graph = ComputationGraph()

    # Build a computation graph
    # Constant parts
    N_const = ConstantNode("N_data", jnp.ones(100))
    F_const = ConstantNode("F_matrix", jnp.ones((100, 10)))
    graph.add_node(N_const)
    graph.add_node(F_const)

    # Variable part
    P_var = VariableNode("P_amplitude")
    graph.add_node(P_var)

    # Intermediate computation: F^T F (constant!)
    FtF = ComputeNode("F^T F", [F_const])
    graph.add_node(FtF)

    # Final computation: N + F^T P F (variable because of P)
    woodbury = ComputeNode("Woodbury", [N_const, FtF, P_var])
    graph.add_node(woodbury)

    graph.mark_output(woodbury)

    print("\nConstant folding analysis:")
    print("-" * 50)
    for node in graph.nodes:
        is_const = getattr(node, "is_constant", False)
        const_str = "CONSTANT" if is_const else "variable"
        print(f"  {node.name:20s} - {const_str}")

    print("\nOptimization:")
    print("  • F^T F can be pre-computed (constant)")
    print("  • Woodbury must be computed each time (depends on variable P)")
    print()


# ============================================================================
# Example: Dead Code Elimination
# ============================================================================


def example_dead_code():
    """
    Example showing elimination of unused intermediate products.

    Sometimes we compute things that aren't actually used in the final result.
    """
    print("\n" + "=" * 70)
    print("Example: Dead Code Elimination")
    print("=" * 70)

    graph = ComputationGraph()

    # Build a more complex structure
    y = GraphNode("y_data")
    N = GraphNode("N_matrix")
    F = GraphNode("F_matrix")
    P = GraphNode("P_matrix")
    graph.add_node(y)
    graph.add_node(N)
    graph.add_node(F)
    graph.add_node(P)

    # Woodbury solve needs N, F, P
    woodbury_solve = GraphNode("woodbury_solve")
    woodbury_solve.add_dependency(N)
    woodbury_solve.add_dependency(F)
    woodbury_solve.add_dependency(P)
    woodbury_solve.add_dependency(y)
    graph.add_node(woodbury_solve)

    # Someone computed the full matrix N + F^T P F (NOT NEEDED!)
    full_matrix = GraphNode("full_matrix_UNUSED")
    full_matrix.add_dependency(N)
    full_matrix.add_dependency(F)
    full_matrix.add_dependency(P)
    graph.add_node(full_matrix)

    # Someone computed just F^T F (NOT NEEDED!)
    FtF = GraphNode("F^T F_UNUSED")
    FtF.add_dependency(F)
    graph.add_node(FtF)

    # Log-likelihood uses woodbury_solve
    loglike = GraphNode("log_likelihood")
    loglike.add_dependency(woodbury_solve)
    graph.add_node(loglike)

    # Mark output
    graph.mark_output(loglike)

    print("\nBefore pruning:")
    print("-" * 50)
    for node in graph.nodes:
        print(f"  {node}")

    # Prune
    removed = graph.prune()

    print("\nAfter pruning:")
    print("-" * 50)
    for node in graph.nodes:
        print(f"  {node}")

    print(f"\nRemoved {removed} dead nodes!")
    print("✓ full_matrix and F^T F were never used, so they're eliminated")
    print()


if __name__ == "__main__":
    example_pruning()
    example_constant_folding()
    example_dead_code()

    print("=" * 70)
    print("Summary: Graph Optimization")
    print("=" * 70)
    print("Three key optimizations:")
    print("  1. Pruning: Remove nodes not contributing to outputs")
    print("  2. Constant folding: Pre-evaluate constant subgraphs")
    print("  3. Dead code elimination: Remove computed-but-unused values")
    print()
    print("These optimizations happen automatically based on graph structure!")
    print("=" * 70)
