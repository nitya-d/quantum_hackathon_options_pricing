# What is a Photonic Quantum Processing Unit?
## Brick 1: Photon generation
Quandela uses quantum dots as the single photon source, which generate individual photons to be used as qubits. A quantum dot is a laser excites electrons to a higher energy level. When the electrons fall back down to its ground state, the quantum dot emits a single photon. 80M photons can be generated per second.
## Brick 2: Quantum interconnects
This is the bridge to take the generated photons and ensure they reach the manipulation and detection module. So this module splits the photons across multiple optical fibers and also delays the photons so they can all enter the photonic chip at the same time. The chip is where the photon manipulation/ computation takes place.
## Brick 3: Manipulation/computation and detection
A photonic integrated chip (PIC) lets the photons interfere with each other and can be programmed to design algorithms. Multiple photons enter the chip on the left. The photons travel along waveguides and get manipulated by beamsplitters  and phase shifters. The photons leave the chip on the right.
**The chip contains two optical elements; beamsplitter and phase shifters. Together they form a linear interferometer and transform the incoming photon states into a superposition of multiple photon states.** The beamsplitter combines two optical modes and lets the photons on each arm to interfere with each other. (These are where the lines cross on the chip.) The phases shifter manipulates only one optical mode at a time.
Once the photons are leaving the photonic chip they are detected by single photon detectors, the output is stored and a output distribution is build, using Boson Sampling.

**Fock states** are a way to count how many photons are in each of several possible modes. A mode is simply a different spatial location, such as a different optical fibre. <br>
**Fock space**: The collection of all possible Fock states for n photons and m modes is called the Fock space. The dimension of this space is given by the binomial coefficient (m+n−1 choose n). This tells us how many different ways we can distribute n photons across m modes. To give some examples this means for 6 photons and 12 optical modes there are 12376 different Fock states.

**Optical circuit**: To manipulate input Fock states, we use linear optical components assembled into what is called an optical circuit. These components are mathematically described as unitary matrices known as scattering matrices. In linear optics, a quantum computation is performed by passing photons through such a circuit. These circuits must preserve the total number of photons.

**Phase shifter**: The first and simplest component is the phase shifter. This component acts on one mode with an angular parameter φ, and its corresponding scattering matrix is (e^iφ), which is a scalar in this case.
How does a phase shifter act on an input Fock state? For a one-photon Fock state, the effect of a phase shifter is: |1⟩⟼eiφ|1⟩. Remember, the state |1⟩ is not a qubit but a Fock state of one photon in one mode.

**Beam splitter**: The second optical component is the beam splitter. As the name suggests, it splits an incoming photon in one mode into a superposition of the photon in two output modes. It is made of a semi-reflective mirror with adjustable transmittance. The beam splitter acts on two modes, and its scattering matrix is described as follows:
```
U = eiϕ0 (sin⁡(θ)eiϕR  cos⁡(θ)e−iϕT
          cos⁡(θ)eiϕT  −sin⁡(θ)e−iϕR)
```
The main parameter of a beam splitter is the angle θ. All other angle values can be adjusted to follow usual conventions.

**Balanced beam splitter**: An interesting special case occurs when θ=π/4. The matrix U becomes:
```
U = eiϕ0 (eiϕR   e−iϕT
          eiϕT  −e−iϕR)
```

**The HOM effect**: The Hong-Ou-Mandel (HOM) effect is a fundamental phenomenon in quantum optics that demonstrates the interference of indistinguishable photons. When two identical photons enter a balanced beam splitter, one in each input mode, they will always exit together in the same output mode due to quantum interference. This behaviour is purely quantum and cannot be explained by classical physics.