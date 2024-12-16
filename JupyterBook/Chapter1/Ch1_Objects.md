# <font color="#0093AF">Objects</font>

## Functional versus object-oriented programming paradigms
### What is functional programming?
Most people are familiar with basic programs that work with data and functions (functional programming paradigm). In functional programming, we have data and functions, and typically one calls a function with data, and new data is returned. For example, suppose we have a density matrix, `rho`, and a propagator, 'U'. We would then likely have a function to propagate the density matrix, e.g. `prop`. Then we would call:

```
rho_new=prop(rho,U)
```

We would also need other functions, for example, a detection matrix and detection function, which would yield signal intensity at some time. Here we assume the signal is stored in a vector, `I`. which we have indexed here with `k`.

```
I[k]=detect(rho,det_mat)
```

Detection and propagation normally would be performed in a *for* loop. 

```
N=1024
I=np.zero(1024)
for k in range(N):
	I[k]=detect(rho,det_mat)
	rho=prop(rho,U)
```

Afterwards, we process the signal, and also calculate a frequency axis. In numpy, this might look like:

```
I[0]/=2
S=np.fft.fftshift(np.fft.fft(I,len(I)*2))
Dt=1e-4
v=1/(2*Dt)*np.linspace(-1,1,len(I)*2)
v-=(v[1]-v[0])/2
```

Finally, we would plot the result.

```
ax=plt.sublots()[1]
ax.plot(v,np.real(S))
ax.set_xlabel(r'$\nu$ / Hz')
```

This is a perfectly reasonable way to program, and powerful simulation programs are based on this principle. In [SPINEVOLUTION](https://spinevolution.com) and [SIMPSON](https://inano.au.dk/about/research-centers-and-projects/nmr/software/simpson), one writes scripts that are input into the program and a file is returned. In these cases, it is less relevant to the user whether objects or functions are used. In [SPINACH](https://spindynamics.org/wiki/index.php?title=Main_Page), one works more interactively and writes code rather than input scripts, and uses the appropriate SPINACH functions to generate output data and figures.

Simulating in SLEEPY is intended to be more interactive than all of these, such that we have direct access to the data being used to run the simulation. A key piece of making this possible, but still manageable for the average user is the application of object-oriented programming.

### What is an object?
Object-oriented programming is based on classes and objects. An object stores both data and functions, and allows one to precisely define how certain types of data are handled (the class defines the object. You can have multiple objects of the same class, each with different data inside). By organizing data with classes, it's easier for the programmer to control what you can and cannot do with the data, reducing mistakes, and we can also provide most of the functions that make sense to apply to a given set of data within the object itself. For example, suppose we have the detection matrix, density operator and propagator from above. In SLEEPY, the steps above can be performed by executing:
```
rho.DetProp(U,n=1024).plot(FT=True)
```
We can do this because `rho` and `U` are objects. `rho` has a function, "DetProp", which performes the detection and propagation functions in sequence *n* times. The detection matrix is stored inside the `rho` object, because we really only need it to detect the density matrix in rho, so there is no reason to carry it around as a separate variable. The function to propagate `rho` by `U` is also stored internally. Indeed, we can even just type `U*rho` to propagate with `U` one time, where objects allow defining the meaning of mathematical symbols; here, this means multiply the propagator matrix by rho and store the new result in rho, where this can include an internal loop over different elements of a powder average. `rho` also stores intensities resulting from detection internally, rather than requiring assigment to an external variable (above, we used `I`), and that also allows `rho` to perform signal processing internally. The "plot" function then creates a figure with the Fourier transformed (optionally apodized) data. 

Usage of objects, then, greatly simplifies the coding required by the user to create complex simulations. On the other hand, the objects give the user direct access to all of the data that was used to create the simulation. This makes SLEEPY much less of a (uh) "black-box", so that one may investigate the components going into a simulation.

## Objects in SLEEPY
The key components of SLEEPY are then all objects. This includes the Experimental System (ExpSys), the Hamiltonian (Hamiltonian), the Liouvillian (Liouvillian), the density matrix (Rho), the powder average (PwdAvg), propagators (Propagator), and pulse sequences (Sequence), as well as other more internal components. Each of these objects will describe itself if typed at a python command line (e.g. iPython, Jupyter Notebook, Google Colab, etc.), and also contains a plotting function to show the critical stored data.

## Complications of object-oriented programming
An object has various *attributes*. Some of these attributes are variables. Some are functions. Some are functions that look like variables (that is, they act like variables, but are obtained via an internal calculation). Some attributes can be edited by the user, but others can't. The object itself may sometimes be indexed, and sometimes it can be called, but not always. 

Compared to functional programming (esp. in programming languages like MatLab), where we almost always put data into the function and get data out, with the original data is unmodified, it can be a little less clear what is going on in object-oriented programming. This is the trade-off for the flexibility of object-oriented programs. Our advice is to start with the tutorial examples, and try SLEEPY in a Jupyter Notebook or iPython console, and see what happens. Once you have the hang of it, you should find that you can simulate dynamic systems with much less effort than with a functional paradigm. We hope, also, that the access that SLEEPY gives you to pieces of the simulation is informative and helpful in understanding more about how dynamics simulations are done.