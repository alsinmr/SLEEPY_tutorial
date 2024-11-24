# <font color="maroon">SLEEPY basics</font>

The first chapter of the tutorial introduces the different "objects", how to create them, how to use them, and how they interact. Most objects in SLEEPY are intended to connect to familiar concepts in magnetic resonance, for example, we have a Hamiltonian object, a Liouvillian object, a propagator object, a pulse-sequence object, and a density matrix object. The way these interact should hopefully be at least somewhat intuitive.

## What is an object?
Most people are familiar with basic programs that work with data and functions (functional programming paradigm). For example, we might have a density matrix (let's call it rho, as used in SLEEPY). In the functional paradigm, rho would be a matrix (or vector). If you wanted to propagate rho, you would need a propagator, U (also a matrix), and you would need to calculate
```
rho_new = U @ rho
```
Furthermore, you would need to repeat the calculation for all elements of a powder average, and have a separate rho and propagator for every element of the powder average. You would potentially also need to multiply rho_new by a detection operator to obtain signal after each propagation step. The propagator itself would likely originate from a propagator function applied to the Hamiltonian or Liouvillian, along with a specified timestep. Simple calculations rapidly become tedious in this paradigm. 

On the other hand, we can work with *objects* (object-oriented programming paradigm). An object contains data, but may also contain code that may be applied to that data. For example, in SLEEPY, rho is not a matrix, but rather an object. It does, indeed, contain a matrix internally that corresponds to the density matrix at a given time. But, it also contains also kinds of other useful information and code. For example, rho also contains the detection matrix or matrices. So, any time one wants to detect signal, one simply executes
```
rho()
```
The () indicates the "call" function of an object. The call function could be anything, but in case of rho, it multiplies the internally stored density matrix with the detection operators and stores the results internally. This can also be a challenge if one is used to functional programming, where functions usually return the result of the calculation. For an object, a function may return the result of a calculation, but it may also simply modify data inside the object. 

The rho object also defines how it interacts with a propagator. For example, we may execute:
```
U*rho
```
This propagates the density matrix by the propagator, just like the operation above (`rho_new=U@rho`), but now the propagation is stored internally. If we have a powder average, the propagator object (U) stores matrices for all elements of the powder average, and rho also stores matrices for every element of the powder average. `U*rho` then multiplies all elements internally, without requiring the user to do anything special. Once propagation and detection is performed over multiple propagation steps, the signal is stored in rho.I. rho also contains plotting functions, for example,
```
rho.plot()
rho.plot(FT=True)
```
returns the time domain and Fourier transformed signals.

While this sounds somewhat complicated at the beginning, as you work through the tutorial, you will find that the amount of end-user coding is vastly reduced by the object-oriented paradigm, and it becomes much easier to properly treat different data types when their interactions are defined by the enclosing objects.