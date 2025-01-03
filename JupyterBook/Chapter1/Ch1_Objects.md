# <font color="#0093AF">Objects</font>

## Functional versus object-oriented programming paradigms
### What is functional programming?
Many scientists are familiar with basic programs that work with data and functions (functional programming). In functional programming, we have data and functions, and typically one calls a function with data, and new data is returned. For example, suppose we have a density matrix, `rho`, and a propagator, 'U'. We would then likely have a function to propagate the density matrix, e.g. `prop`. Then we would call:

```
rho_new=prop(rho,U)
```

We would also need other functions, for example, a detection matrix and detection function, which would yield signal intensity at some time. Here we store the signal in a vector, `I`. which we have indexed here with `k`.

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

Afterwards, we process the signal, and also calculate a frequency axis. Using Python's Numpy module (np), this might look like:

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

This is a perfectly reasonable way to program, and powerful simulation programs are based on this principle. In [SpinEvolution](https://Spinevolution.com) and [SIMPSON](https://inano.au.dk/about/research-centers-and-projects/nmr/software/simpson), one writes files that are input into the program and a file is returned. In these cases, it is less relevant to the user whether objects or functions are used, since the user is not doing any direct coding, just creating input files that follow a specific format to communicate the desired simulation to the program. In [Spinach](https://spindynamics.org/wiki/index.php?title=Main_Page), one is writing the Matlab script (i.e. program) directly, and uses the appropriate Spinach functions in combination with MATLAB functionality to generate output data and figures in the MATLAB workspace.

 Similar to Spinach, one directly codes Python scripts to use SLEEPY. However, simulating in SLEEPY is intended to be interactive. One creates *objects*, which in an interactive Python console (e.g. iPython or Jupyter notebook) remain in the computer memory and can be probed by the user, used for a simulation, and recycled as desired (Spinach also uses some objects, but the user does not usually access these). One also has access to the various arrays and matrices being used to run the simulation and calculate the results. A key component of making this possible, but still manageable for the average user to run a simulation, is the application of object-oriented programming.

### What is an object?
[Object-oriented programming](https://en.wikipedia.org/wiki/Object-oriented_programming) is based on [classes](https://en.wikipedia.org/wiki/Class_(computer_programming)) and [objects](https://en.wikipedia.org/wiki/Object_(computer_science)). An object stores both data and functions, and allows one to precisely define how certain types of data are handled (the class defines the object. You can have multiple objects created from the same class, each with different data inside). By organizing data with objects, it's easier for the programmer to control what the user can and cannot do with the data, reducing mistakes, and the programmer can also provide functions to that make sense to apply to a given set of data. Since the functions and data are both within the object, the user does not need to be particularly careful about putting the right data into the function. For example, suppose we have the detection matrix, density operator and propagator from above. In SLEEPY, the steps above can be performed by executing:
```
rho.DetProp(U,n=1024).plot(FT=True)
```
We can do this because `rho` and `U` are objects. `rho` has a function, "DetProp", which performes the detection and propagation functions in sequence *n* times. The detection matrix is stored inside the `rho` object, because we really only need it to detect the density matrix in rho, so there is no reason to carry it around as a separate variable. The function to propagate `rho` by `U` is also stored internally. Indeed, we can even just type `U*rho` to propagate with `U` one time, where objects allow defining the meaning of mathematical symbols; here, this means multiply the propagator matrix by rho and store the new result in rho, where this can include an internal loop over different elements of a powder average. `rho` also stores intensities resulting from detection internally, rather than requiring assigment to an external variable (above, we used `I`), and that also allows `rho` to perform signal processing internally. The "plot" function then creates a figure with the Fourier transformed (optionally apodized) data. 

Usage of objects, then, greatly simplifies the coding required by the user to create complex simulations. On the other hand, the objects give the user direct access to all of the data that was used to create the simulation. This makes SLEEPY much less of a (uh) "black-box", so that one may investigate the components going into a simulation.

## Objects in SLEEPY
The key components of SLEEPY are all objects. This includes the Experimental System (ExpSys), the Hamiltonian (Hamiltonian), the Liouvillian (Liouvillian), the density matrix (Rho), the powder average (PwdAvg), propagators (Propagator), and pulse sequences (Sequence), as well as other more internal components. Each of these objects will return a description of itself if typed at a Python command line or called with `print(object)`, and also contains a plotting function to show features of the stored data.

## Complications of object-oriented programming
An object has various *attributes*. Some of these attributes are data. Some are functions. Some are functions that look like data (that is, they are returned as data, but are obtained via an internal calculation). Some functions return data, but others just modify the data stored inside the object. Some attributes can be edited by the user, but others can't. The object itself may sometimes be indexed (e.g. `L[5]`), and sometimes it can be called (e.g. `rho()`), but not always. 

Further complicating objects is that they are not static, and they have access to the object that created them. If we create a pulse-sequence object from a Liouvillian in SLEEPY, modifications to the Liouvillian even after the sequence creation will affect the behavior of the sequence. This greatly reduces the amount of code require for some calculations, since, for example if we want to modify exchange rates in the Liouvillian, we don't need to regenerate the sequence every time. On the other hand, this behavior can be a little confusing to a new user.

Compared to functional programming (esp. in programming languages like MatLab), where we almost always put data into the function and get data out, with the original data is unmodified, it can be a little less clear what is going on in object-oriented programming. This is the trade-off for the flexibility of object-oriented programs. Our advice is to start with the tutorial examples, and try SLEEPY in a Jupyter Notebook or iPython console, and see what happens. Once you have the hang of it, you should find that you can simulate dynamic systems with much less effort than with a functional paradigm. We hope, also, that the access that SLEEPY gives you to pieces of the simulation is informative and helpful in understanding more about how dynamics simulations are done.

**A quick tip:** We usually create objects in the following order, where modifications to objects should occur directly after their creation. Following that scheme will avoid errors.

ExpSys(1 or more) : Liouvillian : Sequence : Propagator (optional) -> Apply Sequence or Propagator to Rho

However, we can break this rule in many cases. Modifications to the Liouvillian will carry into the Sequence automatically. Modifications to ExpSys will carry over automatically to new calculations in the Liouvillian, however, the Liouvillian contains a cache, which will not register changes to ExpSys (use `L.clear_cache()` in this case). Modifications to the Liouvillian itself always clear its cache. Caching can be turned off entirely if desired, via `sl.Defaults['cache']=False` (sl is the SLEEPY module). Propagators are usually not affected by upstream changes (however, if, for some reason you create a propagator, but don't use it before modifying the Liouvillian, then it will reflect the modified Liouvillian, since propagators are only calculated when required...if your code has this, you're doing it wrong ;-) ).
