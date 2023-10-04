
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

def unitary_from_circuit(circuit:QuantumCircuit, ordering = "Big endian") -> np.ndarray:
    # get unitary from circuit
    if ordering == "Big endian":
        op = Operator(circuit.reverse_bits())
    else:
        op = Operator(circuit)

    return op.data


def simulate_circuit(circuit:QuantumCircuit):
    # simulate circuit
    print(circuit)
    U = unitary_from_circuit(circuit)

    simulate_unitary(U)

def bitstring_to_array(bitstring, qiskit_ordering = False):
    # convert a bitstring ot dirac array
    if qiskit_ordering:
        bitstring = bistring

    n = len(bitstring)

    arr = np.zeros((2**n,1),dtype=np.complex128)


    # conver bistring to int
    i = int(bitstring,2)

    arr[i] = 1

    return arr


def generate_all_n_length_bitstrings(n):
    # generate all n length bitstrings

    bitstrings = []

    for i in range(2**n):
        bitstring = bin(i)[2:].zfill(n)
        bitstring = bitstring
        bitstrings.append(bitstring)

    return bitstrings

def simulate_unitary(U:np.ndarray):
    # simulate unitary evolution

    n = int(np.log2(len(U)))

    print(f"Simulating unitary of size {n}x{n} :\n{U} ")
    bs = generate_all_n_length_bitstrings(n)
    for b in bs:
        print(f"TESTING {b}")
        res = apply_unitary(b,U)

        print(f"In: {b} Out: {res}")

def array_to_bitstring(arr):
    # convert dirac array to bitstring

    n = int(np.log2(len(arr)))
    print(arr)
    # find index of 1
    i = np.where(arr == 1)[0][0]

    # convert to bitstring
    bitstring = bin(i)[2:].zfill(n)

    return bitstring

def apply_unitary(psi:str,U:np.ndarray):
    # apply unitary to a state vector

    assert len(psi) == int(np.log2(len(U))), psi

    psi = bitstring_to_array(psi)

    arr = np.matmul(U,psi)

    return array_to_bitstring(arr)