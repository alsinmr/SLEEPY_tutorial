# <font color="#0093AF">Objects</font>

## What is an object?
### Functional versus object-oriented programming paradigms
Most people are familiar with basic programs that work with data and functions (functional programming paradigm). For example, we might have a density matrix (let's call it rho, as used in SLEEPY). In the functional paradigm, rho would be a matrix (or vector). If you wanted to propagate rho, you would need a propagator, U (also a matrix), and you would need to calculate
```
rho_new = U @ rho
```
Furthermore, you would need to repeat the calculation for all elements of a powder average, and have a separate rho and propagator for every element of the powder average. You would potentially also need to multiply rho_new by a detection operator to obtain signal after each propagation step. The propagator itself would likely originate from a propagator function applied to the Hamiltonian or Liouvillian, along with a specified timestep. Simple calculations rapidly become tedious in this paradigm. 

On the other hand, we can work with *objects* (object-oriented programming paradigm). An object contains data, but may also contain code that may be applied to that data. For example, in SLEEPY, rho is not a matrix, but rather an object. It does, indeed, contain a matrix internally that corresponds to the density matrix at a given time. But, it also contains also kinds of other useful information and code. For example, rho also contains the detection matrix or matrices. So, any time one wants to detect signal, one simply executes
```
rho()
```
The () indicates the "call" function of an object. The call function could be anything, but in case of rho, it multiplies the internally stored density matrix with the detection operators and stores the results internally. This can also be a challenge if one is used to functional programming, where functions usually return the result of the calculation. For an object, a function may return the result of a calculation, but it may also simply modify data inside the object. In SLEEPY, when a function only modifies internal data, then usually the object itself is returned. This allows the user to string together multiple object functions in a single line.

The rho object also defines how it interacts with a propagator. For example, we may execute:
```
U*rho
```
This propagates the density matrix by the propagator, just like the operation above (`rho_new=U@rho`), but now the propagation is stored internally. If we have a powder average, the propagator object (U) stores matrices for all elements of the powder average, and rho also stores matrices for every element of the powder average. `U*rho` then multiplies all powder elements internally, without requiring the user to do anything special. Once propagation and detection is performed over multiple propagation steps, the signal is stored in rho.I. rho also contains plotting functions, for example,
```
rho.plot()
rho.plot(FT=True)
```
returns plots of the time domain and Fourier transformed signals.

### Complications of object-oriented programming
An object has various *attributes*. Some of these attributes are variables. Some are functions. Some are functions that look like variables (that is, they act like variables, but are obtained via an internal calculation). Some attributes can be edited by the user, but others can't. The object itself may sometimes be indexed, and sometimes it can be called, but not always. 

Compared to functional programming (esp. in programming languages like MatLab), where we almost always put data into the function and get data out, and the original data is unmodified, it can be a little less clear what is going on. This is the trade-off for the flexibility of object-oriented programs. Our advice is to start with the tutorial examples, and also just use SLEEPY in a Jupyter Notebook or iPython console, and just see what happens. Once you have the hang of it, you should find that you can simulate dynamic systems with much less effort than with a functional paradigm. We hope, also, that the access that SLEEPY gives you to pieces of the simulation is informative and helpful in understanding more about how dynamics simulations are done.