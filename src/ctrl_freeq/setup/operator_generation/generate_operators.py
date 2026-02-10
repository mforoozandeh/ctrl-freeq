import torch
import numpy as np


def create_hamiltonian_basis(N):
    # Define basic Pauli matrices and identity once
    X = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Y = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)

    def tensor_product(operators):
        result = operators[0]
        for op in operators[1:]:
            result = np.kron(result, op)
        return result

    def generate_spin_operator(spin_number, operator_type):
        operators = [identity] * N
        operators[spin_number] = {"X": X, "Y": Y, "Z": Z}[operator_type]
        return tensor_product(operators)

    spin_operators = {}

    # Case for a single qubit
    if N == 1:
        spin_operators = {"X_1": X, "Y_1": Y, "Z_1": Z}

    # Case for multiple qubits
    if N > 1:
        for spin in range(N):
            for ax in ["X", "Y", "Z"]:
                spin_operators[f"{ax}_{spin + 1}"] = generate_spin_operator(spin, ax)

    return spin_operators


def create_observable_operators(N):
    # Define basic Pauli matrices and identity once
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)

    def tensor_product(operators):
        result = operators[0]
        for op in operators[1:]:
            result = np.kron(result, op)
        return result

    def generate_spin_operator(spin_number, operator_type):
        operators = [identity] * N
        operators[spin_number] = {"X": X, "Y": Y, "Z": Z}[operator_type]
        return tensor_product(operators)

    spin_operators = {}

    # Case for a single qubit
    if N == 1:
        spin_operators = {"X_1": X, "Y_1": Y, "Z_1": Z}

    # Case for multiple qubits
    if N > 1:
        for spin in range(N):
            for ax in ["X", "Y", "Z"]:
                spin_operators[f"{ax}_{spin + 1}"] = generate_spin_operator(spin, ax)

    return spin_operators


def create_hamiltonian_basis_torch(N, dtype=torch.complex128, device="cpu"):
    # Define basic Pauli matrices in PyTorch
    X = 0.5 * torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    Y = 0.5 * torch.tensor([[0, -1j], [1j, 0]], dtype=dtype)
    Z = 0.5 * torch.tensor([[1, 0], [0, -1]], dtype=dtype)
    identity = torch.eye(2, dtype=dtype)

    D = 2**N  # Dimension of the Hilbert space

    # Initialize the tensor of shape (3, N, D, D)
    op_tensor = torch.zeros((3, N, D, D), dtype=dtype)

    def tensor_product(operators):
        """Returns the tensor product of a list of operators."""
        result = operators[0]
        for op in operators[1:]:
            result = torch.kron(result, op)
        return result

    def generate_spin_operator(spin_number, operator):
        """Generates the tensor product for the specified spin number and operator."""
        operators = [identity] * N
        operators[spin_number] = operator  # Apply the operator to the specified qubit
        return tensor_product(operators)

    # Loop through all qubits and assign the X, Y, Z operators
    for spin in range(N):
        op_tensor[0, spin] = generate_spin_operator(spin, X)
        op_tensor[1, spin] = generate_spin_operator(spin, Y)
        op_tensor[2, spin] = generate_spin_operator(spin, Z)

    return op_tensor.to(device)  # Move to device (CPU or GPU)


def create_observable_operators_torch(N, dtype=torch.complex128, device="cpu"):
    # Define basic Pauli matrices in PyTorch
    X = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype)
    identity = torch.eye(2, dtype=dtype)

    D = 2**N  # Dimension of the Hilbert space

    # Initialize the tensor of shape (3, N, D, D)
    op_tensor = torch.zeros((3, N, D, D), dtype=dtype)

    def tensor_product(operators):
        """Returns the tensor product of a list of operators."""
        result = operators[0]
        for op in operators[1:]:
            result = torch.kron(result, op)
        return result

    def generate_spin_operator(spin_number, operator):
        """Generates the tensor product for the specified spin number and operator."""
        operators = [identity] * N
        operators[spin_number] = operator  # Apply the operator to the specified qubit
        return tensor_product(operators)

    # Loop through all qubits and assign the X, Y, Z operators
    for spin in range(N):
        op_tensor[0, spin] = generate_spin_operator(spin, X)
        op_tensor[1, spin] = generate_spin_operator(spin, Y)
        op_tensor[2, spin] = generate_spin_operator(spin, Z)

    return op_tensor.to(device)  # Move to device (CPU or GPU)


def create_density_matrices(N, directions):
    """
    Create density matrices for an N-qubit system with each qubit prepared along the specified axis.

    Args:
    N (int): Number of qubits.
    directions (list): A list of strings specifying the preparation axis for each qubit.
                         Valid strings are 'X', '-X', 'Y', '-Y', 'Z', '-Z'.

    Returns:
    np.ndarray: The resulting density matrix for the N-qubit system.
    """
    # Define Pauli matrices and identity
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity_matrix = np.eye(2, dtype=complex)

    pauli_matrices = {
        "X": 0.5 * (identity_matrix + X),
        "-X": 0.5 * (identity_matrix - X),
        "Y": 0.5 * (identity_matrix + Y),
        "-Y": 0.5 * (identity_matrix - Y),
        "Z": 0.5 * (identity_matrix + Z),
        "-Z": 0.5 * (identity_matrix - Z),
    }

    def tensor_product(operators):
        result = operators[0]
        for op in operators[1:]:
            result = np.kron(result, op)
        return result

    rho = np.zeros((2**N, 2**N), dtype=complex)

    for i in range(N):
        operators = [identity_matrix] * N
        operators[i] = pauli_matrices[directions[i]]
        rho += tensor_product(operators)

    return rho / np.trace(rho)
