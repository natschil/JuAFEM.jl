var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": "DocTestSetup = :(using JuAFEM)"
},

{
    "location": "index.html#JuAFEM.jl-1",
    "page": "Home",
    "title": "JuAFEM.jl",
    "category": "section",
    "text": "A simple finite element toolbox written in Julia."
},

{
    "location": "index.html#Introduction-1",
    "page": "Home",
    "title": "Introduction",
    "category": "section",
    "text": "JuAFEM is a finite element toolbox that provides functionalities to implement finite element analysis in Julia. The aim is to be general and to keep mathematical abstractions. The main functionalities of the package include:Facilitate integration using different quadrature rules.\nDefine different finite element interpolations.\nEvaluate shape functions, derivatives of shape functions etc. for the different interpolations and quadrature rules.\nEvaluate functions and derivatives in the finite element space.\nGenerate simple grids.\nExport grids and solutions to VTK.The best way to get started with JuAFEM is to look at the documented examples.note: Note\nJuAFEM is still under development. If you find a bug, or have ideas for improvements, feel free to open an issue or make a pull request on the JuAFEM GitHub page."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "To install, simply run the following in the Julia REPL:Pkg.clone(\"https://github.com/KristofferC/JuAFEM.jl\")and then runusing JuAFEMto load the package."
},

{
    "location": "manual/fe_intro.html#",
    "page": "Introduction to FEM",
    "title": "Introduction to FEM",
    "category": "page",
    "text": ""
},

{
    "location": "manual/fe_intro.html#Introduction-to-FEM-1",
    "page": "Introduction to FEM",
    "title": "Introduction to FEM",
    "category": "section",
    "text": "Here we will present a very brief introduction to partial differential equations (PDEs) and to the finite element method (FEM). Perhaps the simplest PDE of all is the (linear) heat equation, also known as the Laplace equation. We will use this equation as a demonstrative example of the method, and demonstrate how we go from the strong format of the equation, to the weak form, and then finally to the discrete FE problem."
},

{
    "location": "manual/fe_intro.html#Strong-format-1",
    "page": "Introduction to FEM",
    "title": "Strong format",
    "category": "section",
    "text": "The strong format of the heat equation may be written as:- mathbfnabla cdot mathbfq(u) = b quad x in Omegawhere u is the unknown temperature field, mathbfq is the heat flux and b is an internal heat source. To complete the system of equations we need boundary conditions. There are different types of boundary conditions, but the most common ones are Dirichlet – which means that the solution u is known at some part of the boundary, and Neumann – which means that the gradient of the solution, mathbfnabla is known. For exampleu = u^mathrmp quad forall mathbfx in Gamma_mathrmD\nmathbfq cdot mathbfn = q^mathrmp quad forall mathbfx in Gamma_mathrmNi.e. the temperature is presribed to u^mathrmp at the Dirichlet part of the boundary, Gamma_mathrmD, and the heat flux is prescribed to q^mathrmp at the Neumann part of the boundary, Gamma_mathrmN.We also need a constitutive equation which links the temperature field, u, to the heat flux, mathbfq. The simplest case is to use Fourier\'s lawmathbfq = -k mathbfnablauwhere k is the conductivity of the material. For simplicity we will consider only constant conductivity k."
},

{
    "location": "manual/fe_intro.html#Weak-format-1",
    "page": "Introduction to FEM",
    "title": "Weak format",
    "category": "section",
    "text": "The solution to the equation above is usually calculated from the corresponding weak format. By multiplying the equation with an arbitrary test function delta u, integrating over the domain and using partial integration we obtain the weak form; Find u in mathbbU s.t.int_Omega mathbfdelta u cdot (k mathbfu) mathrmdOmega =\nint_Gamma_mathrmN delta u q^mathrmp mathrmdGamma +\nint_Omega delta u b mathrmdOmega quad forall delta u in mathbbU^0where mathbbU mathbbU^0 are function spaces with sufficiently regular functions. It can be shown that the solution to the weak form is identical to the solution to the strong format."
},

{
    "location": "manual/fe_intro.html#FE-approximation-1",
    "page": "Introduction to FEM",
    "title": "FE-approximation",
    "category": "section",
    "text": "We now introduce the finite element approximation u_h approx u as a sum of shape functions, N_i and nodal values, a_i. We approximate the test function in the same way (known as the Galerkin method):u_mathrmh = sum_i=1^mathrmN N_i a_iqquad\ndelta u_mathrmh = sum_i=1^mathrmN N_i delta a_iWe may now inserted these approximations in the weak format, which results insum_i^N sum_j^N delta a_i int_Omega mathbfnabla N_i cdot (k cdot mathbfnabla N_j) mathrmdOmega a_j =\nsum_i^N delta a_i int_Gamma N_i q^mathrmp mathrmdGamma +\nsum_i^N delta a_i int_Omega N_i b mathrmdOmegaSince delta u can be chosen arbitrary, the nodal values delta a_i can be chosen arbitrary. Thus, the equation can be written as a linear system of equationsunderlineK underlinea = underlinefwhere underlineK is the (tangent) stiffness matrix, underlinea is the solution vector with the nodal values and underlinef is the force vector. The elements of underlineK and underlinef are given byunderlineK_ij =\n    int_Omega mathbfnablaN_i cdot (k cdot mathbfnablaN_j) mathrmdOmega\n\nunderlinef_i =\n    int_Gamma N_i q^mathrmp mathrmdGamma + int_Omega N_i b mathrmdOmegaThe solution to the system (which in this case is linear) is simply given by inverting the matrix underlineK. We also need to take care of the Dirichlet boundary conditions, by enforcing the correct nodal values a_i to the prescribed values.underlinea = underlineK^-1 underlinef"
},

{
    "location": "manual/fe_intro.html#Implementation-1",
    "page": "Introduction to FEM",
    "title": "Implementation",
    "category": "section",
    "text": "In practice, the shape functions N are only non-zero on parts of the domain Omega. Thus, the integrals are evaluated on sub-domains, called elements or cells. Each cell gives a contribution to the global stiffness matrix and force vector. For a solution of the heat equation, as implemented in JuAFEM, check out this thoroughly commented example."
},

{
    "location": "manual/cell_integration.html#",
    "page": "Cell Integration",
    "title": "Cell Integration",
    "category": "page",
    "text": "DocTestSetup = :(using JuAFEM)"
},

{
    "location": "manual/cell_integration.html#Cell-Integration-1",
    "page": "Cell Integration",
    "title": "Cell Integration",
    "category": "section",
    "text": "Something something FEValues"
},

{
    "location": "manual/cell_integration.html#Numerical-Integration-1",
    "page": "Cell Integration",
    "title": "Numerical Integration",
    "category": "section",
    "text": ""
},

{
    "location": "manual/cell_integration.html#Interpolations-1",
    "page": "Cell Integration",
    "title": "Interpolations",
    "category": "section",
    "text": ""
},

{
    "location": "manual/cell_integration.html#Cell-Values-1",
    "page": "Cell Integration",
    "title": "Cell Values",
    "category": "section",
    "text": ""
},

{
    "location": "manual/degrees_of_freedom.html#",
    "page": "Degrees of Freedom",
    "title": "Degrees of Freedom",
    "category": "page",
    "text": "using JuAFEM"
},

{
    "location": "manual/degrees_of_freedom.html#Degrees-of-Freedom-1",
    "page": "Degrees of Freedom",
    "title": "Degrees of Freedom",
    "category": "section",
    "text": "The distribution and numbering of degrees of freedom (dofs) are handled by the DofHandler. The DofHandler will be used to query information about the dofs. For example we can obtain the dofs for a particular cell, which we need when assembling the system.The DofHandler is based on the grid. Here we create a simple grid with Triangle cells, and then create a DofHandler based on the gridgrid = generate_grid(Triangle, (20, 20))\ndh = DofHandler(grid)\n# hide"
},

{
    "location": "manual/degrees_of_freedom.html#Fields-1",
    "page": "Degrees of Freedom",
    "title": "Fields",
    "category": "section",
    "text": "Before we can distribute the dofs we need to specify fields. A field is simply the unknown function(s) we are solving for. To add a field we need a name (a Symbol) and we also need to specify number of components for the field. Here we add a vector field :u (2 components for a 2D problem) and a scalar field :p.push!(dh, :u, 2)\npush!(dh, :p, 1)\n# hideFinally, when we have added all the fields, we have to close! the DofHandler. When the DofHandler is closed it will traverse the grid and distribute all the dofs for the fields we added.close!(dh)"
},

{
    "location": "manual/degrees_of_freedom.html#Specifying-interpolation-for-a-field-1",
    "page": "Degrees of Freedom",
    "title": "Specifying interpolation for a field",
    "category": "section",
    "text": "In the example above we did not specify which interpolation should be used for our fields :u and :p. By default iso-parametric elements will be used meaning that the interpolation that matches the grid will be used – for a linear grid a linear interpolation will be used etc. It is sometimes useful to separate the grid interpolation from the interpolation that is used to approximate our fields (e.g. sub- and super-parametric elements).We can specify which interpolation that should be used for the approximation when we add the fields to the dofhandler. For example, here we add our vector field :u with a quadratic interpolation, and our :p field with a linear approximation.dh = DofHandler(grid) # hide\npush!(dh, :u, 2, Lagrange{2,RefTetrahedron,2}())\npush!(dh, :p, 1, Lagrange{2,RefTetrahedron,1}())\n# hide"
},

{
    "location": "manual/degrees_of_freedom.html#Ordering-of-Dofs-1",
    "page": "Degrees of Freedom",
    "title": "Ordering of Dofs",
    "category": "section",
    "text": "ordered in the same order as we add to dofhandler nodes -> (edges ->) faces -> cells"
},

{
    "location": "manual/assembly.html#",
    "page": "Assembly",
    "title": "Assembly",
    "category": "page",
    "text": "DocTestSetup = :(using JuAFEM)"
},

{
    "location": "manual/assembly.html#Assembly-1",
    "page": "Assembly",
    "title": "Assembly",
    "category": "section",
    "text": "When the local stiffness matrix and force vector have been calculated they should be assembled into the global stiffness matrix and the global force vector. This is just a matter of adding the local matrix and vector to the global one, at the correct place. Consider e.g. assembling the local stiffness matrix ke and the local force vector fe into the global K and f respectively. These should be assembled into the row/column which corresponds to the degrees of freedom for the cell:K[celldofs, celldofs] += ke\nf[celldofs]           += fewhere celldofs is the vector containing the degrees of freedom for the cell. The method above is very inefficient – it is especially costly to index into the sparse matrix K directly. Therefore we will instead use an Assembler that will help with the assembling of both the global stiffness matrix and the global force vector. It is also often convenient to create the sparse matrix just once, and reuse the allocated matrix. This is useful for e.g. iterative solvers or time dependent problems where the sparse matrix structure, or Sparsity Pattern will stay the same in every iteration/ time step."
},

{
    "location": "manual/assembly.html#Sparsity-Pattern-1",
    "page": "Assembly",
    "title": "Sparsity Pattern",
    "category": "section",
    "text": "Given a DofHandler we can obtain the corresponding sparse matrix by using the create_sparsity_pattern function. This will setup a SparseMatrixCSC with stored values on all the places corresponding to the degree of freedom numbering in the DofHandler. This means that when we assemble into the global stiffness matrix there is no need to change the internal representation of the sparse matrix since the sparse structure will not change.Often the finite element problem is symmetric and will result in a symmetric sparse matrix. This information is often something that the sparse solver can take advantage of. If the solver only needs half the matrix there is no need to assemble both halves. For this purpose there is a create_symmetric_sparsity_pattern function that will only create the upper half of the matrix, and return a Symmetric wrapped SparseMatrixCSC.Given a DofHandler dh we can obtain the (symmetric) sparsity pattern asK = create_sparsity_pattern(dh)\nK = create_symmetric_sparsity_pattern(dh)The returned sparse matrix will be used together with an Assembler, which assembles efficiently into the matrix, without modifying the internal representation."
},

{
    "location": "manual/assembly.html#Assembler-1",
    "page": "Assembly",
    "title": "Assembler",
    "category": "section",
    "text": "Assembling efficiently into the sparse matrix requires some extra workspace. This workspace is allocated in an Assembler. start_assemble is used to create an Assembler:A = start_assemble(K)\nA = start_assemble(K, f)where K is the global stiffness matrix, and f the global force vector. It is optional to give the force vector to the assembler – sometimes there is no need to assemble a global force vector.fdsassemble!(A, celldofs, ke)\nassemble!(A, celldofs, ke, fe)To give a moreK = create_sparsity_pattern(dh)\nf = zeros(ndofs(dh))\nA = start_assemble(K, f)\n\nfor cell in CellIterator(dh)\n    ke, fe = ...\n    assemble!(A, celldofs(cell), ke, fe)\nend"
},

{
    "location": "manual/boundary_conditions.html#",
    "page": "Boundary Conditions",
    "title": "Boundary Conditions",
    "category": "page",
    "text": "DocTestSetup = :(using JuAFEM)"
},

{
    "location": "manual/boundary_conditions.html#Boundary-Conditions-1",
    "page": "Boundary Conditions",
    "title": "Boundary Conditions",
    "category": "section",
    "text": "Every PDE is accompanied with boundary conditions. There are different types of boundary conditions, and they need to be handled in different ways. Below we discuss how to handle the most common ones, Dirichlet and Neumann boundary conditions, and how to do it JuAFEM."
},

{
    "location": "manual/boundary_conditions.html#Dirichlet-Boundary-Conditions-1",
    "page": "Boundary Conditions",
    "title": "Dirichlet Boundary Conditions",
    "category": "section",
    "text": "At a Dirichlet boundary the solution is prescribed to a given value. For the discrete FE-solution this means that there are some degrees of freedom that are fixed. To be able to tell which degrees of freedom we should constrain we need the DofHandler.ch = ConstraintHandler(dh)TBWnote: Examples\nThe following commented examples makes use of Dirichlet boundary conditions:Heat Equation\nTODO"
},

{
    "location": "manual/boundary_conditions.html#Neumann-Boundary-Conditions-1",
    "page": "Boundary Conditions",
    "title": "Neumann Boundary Conditions",
    "category": "section",
    "text": "At the Neumann part of the boundary we know something about the gradient of the solution.As an example, the following code snippet can be included in the element routine, to evaluate the boundary integral:for face in 1:nfaces(cell)\n    if onboundary(cell, face) && (cellid(cell), face) ∈ getfaceset(grid, \"Neumann Boundary\")\n        reinit!(facevalues, cell, face)\n        for q_point in 1:getnquadpoints(facevalues)\n            dΓ = getdetJdV(facevalues, q_point)\n            for i in 1:getnbasefunctions(facevalues)\n                δu = shape_value(facevalues, q_point, i)\n                fe[i] += δu * b * dΓ\n            end\n        end\n    end\nendWe start by looping over all the faces of the cell, next we have to check if this particular face is located on the boundary, and then also check that the face is located on our face-set called \"Neumann Boundary\". If we have determined that the current face is indeed on the boundary and in our faceset, then we reinitialize facevalues for this face, using reinit!. When reinit!ing facevalues we also need to give the face number in addition to the cell. Next we simply loop over the quadrature points of the face, and then loop over all the test functions and assemble the contribution to the force vector.note: Examples\nThe following commented examples makes use of Neumann boundary conditions:TODO"
},

{
    "location": "manual/grid.html#",
    "page": "Grid",
    "title": "Grid",
    "category": "page",
    "text": "DocTestSetup = :(using JuAFEM)"
},

{
    "location": "manual/grid.html#Grid-1",
    "page": "Grid",
    "title": "Grid",
    "category": "section",
    "text": "TODO: Describe the grid format, and how to add sets etc.note: Note\nWrite some conversion functions from e.g. Abaqus file format."
},

{
    "location": "manual/export.html#",
    "page": "Export",
    "title": "Export",
    "category": "page",
    "text": "using JuAFEM\ngrid = generate_grid(Triangle, (2, 2))\ndh = DofHandler(grid); push!(dh, :u, 1); close!(dh)\nu = rand(ndofs(dh)); σ = rand(getncells(grid))"
},

{
    "location": "manual/export.html#Export-1",
    "page": "Export",
    "title": "Export",
    "category": "section",
    "text": "When the problem is solved, and the solution vector u is known we typically want to visualize it. The simplest way to do this is to write the solution to a VTK-file, which can be viewed in e.g. Paraview. To write VTK-files, JuAFEM uses, and extends, functions from the WriteVTK.jl package to simplify the exporting.First we need to create a file, based on the grid. This is done with the vtk_grid function:vtk = vtk_grid(\"my-solution\", grid)\n# hideNext we have to add data to the file. We may add different kinds of data; point data using vtk_point_data or cell data using vtk_cell_data. Point data is data for each nodal coordinate in the grid, for example our solution vector. Point data can be either scalars or vectors. Cell data is – as the name suggests – data for each cell. This can be for example the stress. As an example, lets add a solution vector u as point data, and a vector with stress for each cell, σ, as cell data:vtk_point_data(vtk, u, \"my-point-data\")\nvtk_cell_data(vtk,  σ, \"my-cell-data\")\n# hideFinally, we need to save the file to disk, using vtk_savevtk_save(vtk)\nrm(\"my-solution.vtu\") # hideAlternatively, all of the above can be done using a do block:vtk_grid(\"my-solution\", grid) do vtk\n    vtk_point_data(vtk, u, \"my-point-data\")\n    vtk_cell_data(vtk, σ, \"my-cell-data\")\nend\nrm(\"my-solution.vtu\") # hideFor other functionality, and more information refer to the WriteVTK.jl README. In particular, for exporting the solution at multiple time steps, the section on PVD files is useful."
},

{
    "location": "manual/export.html#Exporting-with-DofHandler-1",
    "page": "Export",
    "title": "Exporting with DofHandler",
    "category": "section",
    "text": "There is an even more convenient way to export a solution vector u – using the DofHandler. The DofHandler already contains all of the information needed, such as the names of our fields and if they are scalar or vector fields. But most importantly the DofHandler knows about the numbering and distribution of degrees of freedom, and thus knows how to \"distribute\" the solution vector on the grid. For example, lets say we have a DofHandler dh and a solution vector u:vtk = vtk_grid(\"my-solution\", dh)\nvtk_point_data(vtk, dh, u)\nvtk_save(vtk)\nrm(\"my-solution.vtu\") # hideor with a do-block:vtk_grid(\"my-solution\", dh) do vtk\n    vtk_point_data(vtk, dh, u)\n    vtk_cell_data(vtk, σ, \"my-cell-data\")\nend\nrm(\"my-solution.vtu\") # hideWhen vtk_point_data is used with a DofHandler all of the fields will be written to the VTK file, and the names will be determined by the fieldname symbol that was used when the field was added to the DofHandler."
},

{
    "location": "manual/export.html#Exporting-Boundary-Conditions-1",
    "page": "Export",
    "title": "Exporting Boundary Conditions",
    "category": "section",
    "text": "There is also a vtk_point_data which accepts a ConstraintHandler. This method is useful to verify that the boundary conditions are applied where they are supposed to. For a ConstraintHandler ch we can export the boundary conditions asvtk_grid(\"boundary-conditions\", grid) do vtk\n    vtk_point_data(vtk, ch)\nendThis will export zero-valued fields with ones on the parts where the boundary conditions are active."
},

{
    "location": "examples/generated/heat_equation.html#",
    "page": "Heat Equation",
    "title": "Heat Equation",
    "category": "page",
    "text": "EditURL = \"https://github.com/KristofferC/JuAFEM.jl/blob/master/docs/src/examples/heat_equation.jl\""
},

{
    "location": "examples/generated/heat_equation.html#Heat-Equation-1",
    "page": "Heat Equation",
    "title": "Heat Equation",
    "category": "section",
    "text": "(Image: )tip: Tip\nThis example is also available as a Jupyter notebook: heat_equation.ipynb"
},

{
    "location": "examples/generated/heat_equation.html#Introduction-1",
    "page": "Heat Equation",
    "title": "Introduction",
    "category": "section",
    "text": "The heat equation is the \"Hello, world!\" equation of finite elements. Here we solve the equation on a unit square, with a uniform internal source. The strong form of the (linear) heat equation is given by- nabla cdot (k nabla u) = f  quad x in Omegawhere u is the unknown temperature field, k the heat conductivity, f the heat source and Omega the domain. For simplicity we set f = 1 and k = 1. We will consider homogeneous Dirichlet boundary conditions such thatu(x) = 0 quad x in partial Omegawhere partial Omega denotes the boundary of Omega.The resulting weak form is given byint_Omega nabla v cdot nabla u  dOmega = int_Omega v  dOmegawhere v is a suitable test function."
},

{
    "location": "examples/generated/heat_equation.html#Commented-Program-1",
    "page": "Heat Equation",
    "title": "Commented Program",
    "category": "section",
    "text": "Now we solve the problem in JuAFEM. What follows is a program spliced with comments. The full program, without comments, can be found in the next section.First we load the JuAFEM package.using JuAFEMWe start  generating a simple grid with 20x20 quadrilateral elements using generate_grid. The generator defaults to the unit square, so we don\'t need to specify the corners of the domain.grid = generate_grid(Quadrilateral, (20, 20));"
},

{
    "location": "examples/generated/heat_equation.html#Trial-and-test-functions-1",
    "page": "Heat Equation",
    "title": "Trial and test functions",
    "category": "section",
    "text": "A CellValues facilitates the process of evaluating values and gradients of test and trial functions (among other things). Since the problem is a scalar problem we will use a CellScalarValues object. To define this we need to specify an interpolation space for the shape functions. We use Lagrange functions (both for interpolating the function and the geometry) based on the reference \"cube\". We also define a quadrature rule based on the same reference cube. We combine the interpolation and the quadrature rule to a CellScalarValues object.dim = 2\nip = Lagrange{dim, RefCube, 1}()\nqr = QuadratureRule{dim, RefCube}(2)\ncellvalues = CellScalarValues(qr, ip);"
},

{
    "location": "examples/generated/heat_equation.html#Degrees-of-freedom-1",
    "page": "Heat Equation",
    "title": "Degrees of freedom",
    "category": "section",
    "text": "Next we need to define a DofHandler, which will take care of numbering and distribution of degrees of freedom for our approximated fields. We create the DofHandler and then add a single field called u. Lastly we close! the DofHandler, it is now that the dofs are distributed for all the elements.dh = DofHandler(grid)\npush!(dh, :u, 1)\nclose!(dh);Now that we have distributed all our dofs we can create our tangent matrix, using create_sparsity_pattern. This function returns a sparse matrix with the correct elements stored.K = create_sparsity_pattern(dh);We can inspect the pattern using the spy function from UnicodePlots.jl. By default the stored values are set to 0, so we first need to fill the stored values, e.g. K.nzval with something meaningful.using UnicodePlots\nfill!(K.nzval, 1.0)\nspy(K)"
},

{
    "location": "examples/generated/heat_equation.html#Boundary-conditions-1",
    "page": "Heat Equation",
    "title": "Boundary conditions",
    "category": "section",
    "text": "In JuAFEM constraints like Dirichlet boundary conditions are handled by a ConstraintHandler.ch = ConstraintHandler(dh);Next we need to add constraints to ch. For this problem we define homogeneous Dirichlet boundary conditions on the whole boundary, i.e. the union of all the face sets on the boundary.∂Ω = union(getfaceset.(grid, [\"left\", \"right\", \"top\", \"bottom\"])...);Now we are set up to define our constraint. We specify which field the condition is for, and our combined face set ∂Ω. The last argument is a function which takes the spatial coordinate x and the current time t and returns the prescribed value. In this case it is trivial – no matter what x and t we return 0. When we have specified our constraint we add! it to ch.dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)\nadd!(ch, dbc);We also need to close! and update! our boundary conditions. When we call close! the dofs which will be constrained by the boundary conditions are calculated and stored in our ch object. Since the boundary conditions are, in this case, independent of time we can update! them directly with e.g. t = 0.close!(ch)\nupdate!(ch, 0.0);"
},

{
    "location": "examples/generated/heat_equation.html#Assembling-the-linear-system-1",
    "page": "Heat Equation",
    "title": "Assembling the linear system",
    "category": "section",
    "text": "Now we have all the pieces needed to assemble the linear system, K u = f. We define a function, doassemble to do the assembly, which takes our cellvalues, the sparse matrix and our DofHandler as input arguments. The function returns the assembled stiffness matrix, and the force vector.function doassemble(cellvalues::CellScalarValues{dim}, K::SparseMatrixCSC, dh::DofHandler) where {dim}\n    #\' We allocate the element stiffness matrix and element force vector\n    #\' just once before looping over all the cells instead of allocating\n    #\' them every time in the loop.\n    n_basefuncs = getnbasefunctions(cellvalues)\n    Ke = zeros(n_basefuncs, n_basefuncs)\n    fe = zeros(n_basefuncs)\n\n    #\' Next we define the global force vector `f` and use that and\n    #\' the stiffness matrix `K` and create an assembler. The assembler\n    #\' is just a thin wrapper around `f` and `K` and some extra storage\n    #\' to make the assembling faster.\n    f = zeros(ndofs(dh))\n    assembler = start_assemble(K, f)\n\n    #\' It is now time to loop over all the cells in our grid. We do this by iterating\n    #\' over a `CellIterator`. The iterator caches some useful things for us, for example\n    #\' the nodal coordinates for the cell, and the local degrees of freedom.\n    @inbounds for cell in CellIterator(dh)\n        #\' Always remember to reset the element stiffness matrix and\n        #\' force vector since we reuse them for all elements.\n        fill!(Ke, 0)\n        fill!(fe, 0)\n\n        #\' For each cell we also need to reinitialize the cached values in `cellvalues`.\n        reinit!(cellvalues, cell)\n\n        #\' It is now time to loop over all the quadrature points in the cell and\n        #\' assemble the contribution to `Ke` and `fe`. The integration weight\n        #\' can be queried from `cellvalues` by `getdetJdV`.\n        for q_point in 1:getnquadpoints(cellvalues)\n            dΩ = getdetJdV(cellvalues, q_point)\n            #\' For each quadrature point we loop over all the (local) shape functions.\n            #\' We need the value and gradient of the testfunction `v` and also the gradient\n            #\' of the trial function `u`. We get all of these from `cellvalues`.\n            for i in 1:n_basefuncs\n                v  = shape_value(cellvalues, q_point, i)\n                ∇v = shape_gradient(cellvalues, q_point, i)\n                fe[i] += v * dΩ\n                for j in 1:n_basefuncs\n                    ∇u = shape_gradient(cellvalues, q_point, j)\n                    Ke[i, j] += (∇v ⋅ ∇u) * dΩ\n                end\n            end\n        end\n\n        #\' The last step in the element loop is to assemble `Ke` and `fe`\n        #\' into the global `K` and `f` with `assemble!`.\n        assemble!(assembler, celldofs(cell), fe, Ke)\n    end\n    return K, f\nend\nnothing # hide"
},

{
    "location": "examples/generated/heat_equation.html#Solution-of-the-system-1",
    "page": "Heat Equation",
    "title": "Solution of the system",
    "category": "section",
    "text": "The last step is to solve the system. First we call doassemble to obtain the global stiffness matrix K and force vector f.K, f = doassemble(cellvalues, K, dh);To account for the boundary conditions we use the apply! function. This modifies elements in K and f respectively, such that we can get the correct solution vector u by using \\.apply!(K, f, ch)\nu = K \\ f;"
},

{
    "location": "examples/generated/heat_equation.html#Exporting-to-VTK-1",
    "page": "Heat Equation",
    "title": "Exporting to VTK",
    "category": "section",
    "text": "To visualize the result we export the grid and our field u to a VTK-file, which can be viewed in e.g. ParaView.vtk_grid(\"heat_equation\", dh) do vtk\n    vtk_point_data(vtk, dh, u)\nend\nrm(\"heat_equation.vtu\") # hide\nnothing # hide"
},

{
    "location": "examples/generated/heat_equation.html#heat_equation-plain-program-1",
    "page": "Heat Equation",
    "title": "Plain Program",
    "category": "section",
    "text": "Below follows a version of the program without any comments. The file is also available here: heat_equation.jlusing JuAFEM\n\ngrid = generate_grid(Quadrilateral, (20, 20));\n\ndim = 2\nip = Lagrange{dim, RefCube, 1}()\nqr = QuadratureRule{dim, RefCube}(2)\ncellvalues = CellScalarValues(qr, ip);\n\ndh = DofHandler(grid)\npush!(dh, :u, 1)\nclose!(dh);\n\nK = create_sparsity_pattern(dh);\n\nusing UnicodePlots\nfill!(K.nzval, 1.0)\nspy(K)\n\nch = ConstraintHandler(dh);\n\n∂Ω = union(getfaceset.(grid, [\"left\", \"right\", \"top\", \"bottom\"])...);\n\ndbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)\nadd!(ch, dbc);\n\nclose!(ch)\nupdate!(ch, 0.0);\n\nfunction doassemble(cellvalues::CellScalarValues{dim}, K::SparseMatrixCSC, dh::DofHandler) where {dim}\n\n    n_basefuncs = getnbasefunctions(cellvalues)\n    Ke = zeros(n_basefuncs, n_basefuncs)\n    fe = zeros(n_basefuncs)\n\n    f = zeros(ndofs(dh))\n    assembler = start_assemble(K, f)\n\n    @inbounds for cell in CellIterator(dh)\n\n        fill!(Ke, 0)\n        fill!(fe, 0)\n\n        reinit!(cellvalues, cell)\n\n        for q_point in 1:getnquadpoints(cellvalues)\n            dΩ = getdetJdV(cellvalues, q_point)\n\n            for i in 1:n_basefuncs\n                v  = shape_value(cellvalues, q_point, i)\n                ∇v = shape_gradient(cellvalues, q_point, i)\n                fe[i] += v * dΩ\n                for j in 1:n_basefuncs\n                    ∇u = shape_gradient(cellvalues, q_point, j)\n                    Ke[i, j] += (∇v ⋅ ∇u) * dΩ\n                end\n            end\n        end\n\n        assemble!(assembler, celldofs(cell), fe, Ke)\n    end\n    return K, f\nend\n\nK, f = doassemble(cellvalues, K, dh);\n\napply!(K, f, ch)\nu = K \\ f;\n\nvtk_grid(\"heat_equation\", dh) do vtk\n    vtk_point_data(vtk, dh, u)\nend"
},

{
    "location": "reference/quadrature.html#",
    "page": "Quadrature",
    "title": "Quadrature",
    "category": "page",
    "text": "CurrentModule = JuAFEM\nDocTestSetup = :(using JuAFEM)"
},

{
    "location": "reference/quadrature.html#JuAFEM.QuadratureRule",
    "page": "Quadrature",
    "title": "JuAFEM.QuadratureRule",
    "category": "type",
    "text": "QuadratureRule{dim,shape}([quad_rule_type::Symbol], order::Int)\n\nCreate a QuadratureRule used for integration. dim is the space dimension, shape an AbstractRefShape and order the order of the quadrature rule. quad_rule_type is an optional argument determining the type of quadrature rule, currently the :legendre and :lobatto rules are implemented.\n\nA QuadratureRule is used to approximate an integral on a domain by a weighted sum of function values at specific points:\n\nintlimits_Omega f(mathbfx) textd Omega approx sumlimits_q = 1^n_q f(mathbfx_q) w_q\n\nThe quadrature rule consists of n_q points in space mathbfx_q with corresponding weights w_q.\n\nIn JuAFEM, the QuadratureRule type is mostly used as one of the components to create a CellValues or FaceValues object.\n\nCommon methods:\n\ngetpoints : the points of the quadrature rule\ngetweights : the weights of the quadrature rule\n\nExample:\n\njulia> QuadratureRule{2, RefTetrahedron}(1)\nJuAFEM.QuadratureRule{2,JuAFEM.RefTetrahedron,Float64}([0.5], Tensors.Tensor{1,2,Float64,2}[[0.333333, 0.333333]])\n\njulia> QuadratureRule{1, RefCube}(:lobatto, 2)\nJuAFEM.QuadratureRule{1,JuAFEM.RefCube,Float64}([1.0, 1.0], Tensors.Tensor{1,1,Float64,1}[[-1.0], [1.0]])\n\n\n\n"
},

{
    "location": "reference/quadrature.html#JuAFEM.AbstractRefShape",
    "page": "Quadrature",
    "title": "JuAFEM.AbstractRefShape",
    "category": "type",
    "text": "Represents a reference shape which quadrature rules and interpolations are defined on. Currently, the only concrete types that subtype this type are RefCube in 1, 2 and 3 dimensions, and RefTetrahedron in 2 and 3 dimensions.\n\n\n\n"
},

{
    "location": "reference/quadrature.html#JuAFEM.getpoints",
    "page": "Quadrature",
    "title": "JuAFEM.getpoints",
    "category": "function",
    "text": "getpoints(qr::QuadratureRule)\n\nReturn the points of the quadrature rule.\n\nExamples\n\njulia> qr = QuadratureRule{2, RefTetrahedron}(:legendre, 2);\n\njulia> getpoints(qr)\n3-element Array{Tensors.Tensor{1,2,Float64,2},1}:\n [0.166667, 0.166667]\n [0.166667, 0.666667]\n [0.666667, 0.166667]\n\n\n\n"
},

{
    "location": "reference/quadrature.html#JuAFEM.getweights",
    "page": "Quadrature",
    "title": "JuAFEM.getweights",
    "category": "function",
    "text": "getweights(qr::QuadratureRule)\n\nReturn the weights of the quadrature rule.\n\nExamples\n\njulia> qr = QuadratureRule{2, RefTetrahedron}(:legendre, 2);\n\njulia> getweights(qr)\n3-element Array{Float64,1}:\n 0.166667\n 0.166667\n 0.166667\n\n\n\n"
},

{
    "location": "reference/quadrature.html#Quadrature-1",
    "page": "Quadrature",
    "title": "Quadrature",
    "category": "section",
    "text": "QuadratureRule\nAbstractRefShape\ngetpoints\ngetweights"
},

{
    "location": "reference/interpolations.html#",
    "page": "Interpolation",
    "title": "Interpolation",
    "category": "page",
    "text": "CurrentModule = JuAFEM\nDocTestSetup = :(using JuAFEM)"
},

{
    "location": "reference/interpolations.html#JuAFEM.Interpolation",
    "page": "Interpolation",
    "title": "JuAFEM.Interpolation",
    "category": "type",
    "text": "Interpolation{dim, ref_shape, order}()\n\nReturn an Interpolation of given dimension dim, reference shape (see see AbstractRefShape) ref_shape and order order. order corresponds to the highest order term in the polynomial. The interpolation is used to define shape functions to interpolate a function between nodes.\n\nThe following interpolations are implemented:\n\nLagrange{1,RefCube,1}\nLagrange{1,RefCube,2}\nLagrange{2,RefCube,1}\nLagrange{2,RefCube,2}\nLagrange{2,RefTetrahedron,1}\nLagrange{2,RefTetrahedron,2}\nLagrange{3,RefCube,1}\nSerendipity{2,RefCube,2}\nLagrange{3,RefTetrahedron,1}\nLagrange{3,RefTetrahedron,2}\n\nExamples\n\njulia> ip = Lagrange{2,RefTetrahedron,2}()\nJuAFEM.Lagrange{2,JuAFEM.RefTetrahedron,2}()\n\njulia> getnbasefunctions(ip)\n6\n\n\n\n"
},

{
    "location": "reference/interpolations.html#JuAFEM.getnbasefunctions",
    "page": "Interpolation",
    "title": "JuAFEM.getnbasefunctions",
    "category": "function",
    "text": "Return the number of base functions for an Interpolation or Values object.\n\n\n\n"
},

{
    "location": "reference/interpolations.html#JuAFEM.getdim",
    "page": "Interpolation",
    "title": "JuAFEM.getdim",
    "category": "function",
    "text": "Return the dimension of an Interpolation\n\n\n\n"
},

{
    "location": "reference/interpolations.html#JuAFEM.getrefshape",
    "page": "Interpolation",
    "title": "JuAFEM.getrefshape",
    "category": "function",
    "text": "Return the reference shape of an Interpolation\n\n\n\n"
},

{
    "location": "reference/interpolations.html#JuAFEM.getorder",
    "page": "Interpolation",
    "title": "JuAFEM.getorder",
    "category": "function",
    "text": "Return the polynomial order of the Interpolation\n\n\n\n"
},

{
    "location": "reference/interpolations.html#reference-interpolation-1",
    "page": "Interpolation",
    "title": "Interpolation",
    "category": "section",
    "text": "Interpolation\ngetnbasefunctions\ngetdim\ngetrefshape\ngetorder"
},

{
    "location": "reference/fevalues.html#",
    "page": "FEValues",
    "title": "FEValues",
    "category": "page",
    "text": "CurrentModule = JuAFEM\nDocTestSetup = :(using JuAFEM)"
},

{
    "location": "reference/fevalues.html#FEValues-1",
    "page": "FEValues",
    "title": "FEValues",
    "category": "section",
    "text": ""
},

{
    "location": "reference/fevalues.html#JuAFEM.CellValues",
    "page": "FEValues",
    "title": "JuAFEM.CellValues",
    "category": "type",
    "text": "CellScalarValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])\nCellVectorValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])\n\nA CellValues object facilitates the process of evaluating values of shape functions, gradients of shape functions, values of nodal functions, gradients and divergences of nodal functions etc. in the finite element cell. There are two different types of CellValues: CellScalarValues and CellVectorValues. As the names suggest, CellScalarValues utilizes scalar shape functions and CellVectorValues utilizes vectorial shape functions. For a scalar field, the CellScalarValues type should be used. For vector field, both subtypes can be used.\n\nArguments:\n\nT: an optional argument (default to Float64) to determine the type the internal data is stored as.\nquad_rule: an instance of a QuadratureRule\nfunc_interpol: an instance of an Interpolation used to interpolate the approximated function\ngeom_interpol: an optional instance of a Interpolation which is used to interpolate the geometry\n\nCommon methods:\n\nreinit!\ngetnquadpoints\ngetdetJdV\nshape_value\nshape_gradient\nshape_symmetric_gradient\nshape_divergence\nfunction_value\nfunction_gradient\nfunction_symmetric_gradient\nfunction_divergence\nspatial_coordinate\n\n\n\n"
},

{
    "location": "reference/fevalues.html#JuAFEM.reinit!",
    "page": "FEValues",
    "title": "JuAFEM.reinit!",
    "category": "function",
    "text": "reinit!(cv::CellValues, x::Vector)\nreinit!(bv::FaceValues, x::Vector, face::Int)\n\nUpdate the CellValues/FaceValues object for a cell or face with coordinates x. The derivatives of the shape functions, and the new integration weights are computed.\n\n\n\n"
},

{
    "location": "reference/fevalues.html#JuAFEM.getnquadpoints",
    "page": "FEValues",
    "title": "JuAFEM.getnquadpoints",
    "category": "function",
    "text": "getnquadpoints(fe_v::Values)\n\nReturn the number of quadrature points for the Values object.\n\n\n\n"
},

{
    "location": "reference/fevalues.html#JuAFEM.getdetJdV",
    "page": "FEValues",
    "title": "JuAFEM.getdetJdV",
    "category": "function",
    "text": "getdetJdV(fe_v::Values, q_point::Int)\n\nReturn the product between the determinant of the Jacobian and the quadrature point weight for the given quadrature point: det(J(mathbfx)) w_q\n\nThis value is typically used when one integrates a function on a finite element cell or face as\n\nintlimits_Omega f(mathbfx) d Omega approx sumlimits_q = 1^n_q f(mathbfx_q) det(J(mathbfx)) w_q intlimits_Gamma f(mathbfx) d Gamma approx sumlimits_q = 1^n_q f(mathbfx_q) det(J(mathbfx)) w_q\n\n\n\n"
},

{
    "location": "reference/fevalues.html#JuAFEM.shape_value",
    "page": "FEValues",
    "title": "JuAFEM.shape_value",
    "category": "function",
    "text": "shape_value(fe_v::Values, q_point::Int, base_function::Int)\n\nReturn the value of shape function base_function evaluated in quadrature point q_point.\n\n\n\n"
},

{
    "location": "reference/fevalues.html#JuAFEM.shape_gradient",
    "page": "FEValues",
    "title": "JuAFEM.shape_gradient",
    "category": "function",
    "text": "shape_gradient(fe_v::Values, q_point::Int, base_function::Int)\n\nReturn the gradient of shape function base_function evaluated in quadrature point q_point.\n\n\n\n"
},

{
    "location": "reference/fevalues.html#JuAFEM.shape_symmetric_gradient",
    "page": "FEValues",
    "title": "JuAFEM.shape_symmetric_gradient",
    "category": "function",
    "text": "shape_symmetric_gradient(fe_v::Values, q_point::Int, base_function::Int)\n\nReturn the symmetric gradient of shape function base_function evaluated in quadrature point q_point.\n\n\n\n"
},

{
    "location": "reference/fevalues.html#JuAFEM.shape_divergence",
    "page": "FEValues",
    "title": "JuAFEM.shape_divergence",
    "category": "function",
    "text": "shape_divergence(fe_v::Values, q_point::Int, base_function::Int)\n\nReturn the divergence of shape function base_function evaluated in quadrature point q_point.\n\n\n\n"
},

{
    "location": "reference/fevalues.html#JuAFEM.function_value",
    "page": "FEValues",
    "title": "JuAFEM.function_value",
    "category": "function",
    "text": "function_value(fe_v::Values, q_point::Int, u::AbstractVector)\n\nCompute the value of the function in a quadrature point. u is a vector with values for the degrees of freedom. For a scalar valued function, u contains scalars. For a vector valued function, u can be a vector of scalars (for use of VectorValues) or u can be a vector of Vecs (for use with ScalarValues).\n\nThe value of a scalar valued function is computed as u(mathbfx) = sumlimits_i = 1^n N_i (mathbfx) u_i where u_i are the value of u in the nodes. For a vector valued function the value is calculated as mathbfu(mathbfx) = sumlimits_i = 1^n N_i (mathbfx) mathbfu_i where mathbfu_i are the nodal values of mathbfu.\n\n\n\n"
},

{
    "location": "reference/fevalues.html#JuAFEM.function_gradient",
    "page": "FEValues",
    "title": "JuAFEM.function_gradient",
    "category": "function",
    "text": "function_scalar_gradient(fe_v::Values{dim}, q_point::Int, u::AbstractVector)\n\nCompute the gradient of the function in a quadrature point. u is a vector with values for the degrees of freedom. For a scalar valued function, u contains scalars. For a vector valued function, u can be a vector of scalars (for use of VectorValues) or u can be a vector of Vecs (for use with ScalarValues).\n\nThe gradient of a scalar function is computed as mathbfnabla u(mathbfx) = sumlimits_i = 1^n mathbfnabla N_i (mathbfx) u_i where u_i are the nodal values of the function. For a vector valued function the gradient is computed as mathbfnabla mathbfu(mathbfx) = sumlimits_i = 1^n mathbfnabla N_i (mathbfx) otimes mathbfu_i where mathbfu_i are the nodal values of mathbfu.\n\n\n\n"
},

{
    "location": "reference/fevalues.html#JuAFEM.function_symmetric_gradient",
    "page": "FEValues",
    "title": "JuAFEM.function_symmetric_gradient",
    "category": "function",
    "text": "function_symmetric_gradient(fe_v::Values, q_point::Int, u::AbstractVector)\n\nCompute the symmetric gradient of the function, see function_gradient. Return a SymmetricTensor.\n\nThe symmetric gradient of a scalar function is computed as left mathbfnabla  mathbfu(mathbfx_q) right^textsym =  sumlimits_i = 1^n  frac12 left mathbfnabla N_i (mathbfx_q) otimes mathbfu_i + mathbfu_i  otimes  mathbfnabla N_i (mathbfx_q) right where mathbfu_i are the nodal values of the function.\n\n\n\n"
},

{
    "location": "reference/fevalues.html#JuAFEM.function_divergence",
    "page": "FEValues",
    "title": "JuAFEM.function_divergence",
    "category": "function",
    "text": "function_divergence(fe_v::Values, q_point::Int, u::AbstractVector)\n\nCompute the divergence of the vector valued function in a quadrature point.\n\nThe divergence of a vector valued functions in the quadrature point mathbfx_q) is computed as mathbfnabla cdot mathbfu(mathbfx_q) = sumlimits_i = 1^n mathbfnabla N_i (mathbfx_q) cdot mathbfu_i where mathbfu_i are the nodal values of the function.\n\n\n\n"
},

{
    "location": "reference/fevalues.html#JuAFEM.spatial_coordinate",
    "page": "FEValues",
    "title": "JuAFEM.spatial_coordinate",
    "category": "function",
    "text": "spatial_coordinate(fe_v::Values{dim}, q_point::Int, x::AbstractVector)\n\nCompute the spatial coordinate in a quadrature point. x contains the nodal coordinates of the cell.\n\nThe coordinate is computed, using the geometric interpolation, as mathbfx = sumlimits_i = 1^n M_i (mathbfx) mathbfhatx_i\n\n\n\n"
},

{
    "location": "reference/fevalues.html#reference-cellvalues-1",
    "page": "FEValues",
    "title": "CellValues",
    "category": "section",
    "text": "CellValues\nreinit!\ngetnquadpoints\ngetdetJdV\n\nshape_value\nshape_gradient\nshape_symmetric_gradient\nshape_divergence\n\nfunction_value\nfunction_gradient\nfunction_symmetric_gradient\nfunction_divergence\nspatial_coordinate"
},

{
    "location": "reference/fevalues.html#JuAFEM.FaceValues",
    "page": "FEValues",
    "title": "JuAFEM.FaceValues",
    "category": "type",
    "text": "FaceScalarValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])\nFaceVectorValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])\n\nA FaceValues object facilitates the process of evaluating values of shape functions, gradients of shape functions, values of nodal functions, gradients and divergences of nodal functions etc. on the faces of finite elements. There are two different types of FaceValues: FaceScalarValues and FaceVectorValues. As the names suggest, FaceScalarValues utilizes scalar shape functions and FaceVectorValues utilizes vectorial shape functions. For a scalar field, the FaceScalarValues type should be used. For vector field, both subtypes can be used.\n\nnote: Note\nThe quadrature rule for the face should be given with one dimension lower. I.e. for a 3D case, the quadrature rule should be in 2D.\n\nArguments:\n\nT: an optional argument to determine the type the internal data is stored as.\nquad_rule: an instance of a QuadratureRule\nfunc_interpol: an instance of an Interpolation used to interpolate the approximated function\ngeom_interpol: an optional instance of an Interpolation which is used to interpolate the geometry\n\nCommon methods:\n\nreinit!\ngetnquadpoints\ngetdetJdV\nshape_value\nshape_gradient\nshape_symmetric_gradient\nshape_divergence\nfunction_value\nfunction_gradient\nfunction_symmetric_gradient\nfunction_divergence\nspatial_coordinate\n\n\n\n"
},

{
    "location": "reference/fevalues.html#JuAFEM.getcurrentface",
    "page": "FEValues",
    "title": "JuAFEM.getcurrentface",
    "category": "function",
    "text": "getcurrentface(fv::FaceValues)\n\nReturn the current active face of the FaceValues object (from last reinit!).\n\n\n\n"
},

{
    "location": "reference/fevalues.html#reference-facevalues-1",
    "page": "FEValues",
    "title": "FaceValues",
    "category": "section",
    "text": "All of the methods for CellValues apply for FaceValues as well. In addition, there are some methods that are unique for FaecValues:FaceValues\ngetcurrentface"
},

{
    "location": "reference/dofhandler.html#",
    "page": "DofHandler",
    "title": "DofHandler",
    "category": "page",
    "text": "DocTestSetup = :(using JuAFEM)"
},

{
    "location": "reference/dofhandler.html#JuAFEM.DofHandler",
    "page": "DofHandler",
    "title": "JuAFEM.DofHandler",
    "category": "type",
    "text": "DofHandler(grid::Grid)\n\nConstruct a DofHandler based on the grid grid.\n\n\n\n"
},

{
    "location": "reference/dofhandler.html#DofHandler-1",
    "page": "DofHandler",
    "title": "DofHandler",
    "category": "section",
    "text": "DofHandler"
},

{
    "location": "reference/assembly.html#",
    "page": "Assembly",
    "title": "Assembly",
    "category": "page",
    "text": "DocTestSetup = :(using JuAFEM)"
},

{
    "location": "reference/assembly.html#JuAFEM.start_assemble",
    "page": "Assembly",
    "title": "JuAFEM.start_assemble",
    "category": "function",
    "text": "start_assemble([N=0]) -> Assembler\n\nCall before starting an assembly.\n\nReturns an Assembler type that is used to hold the intermediate data before an assembly is finished.\n\n\n\n"
},

{
    "location": "reference/assembly.html#JuAFEM.assemble!",
    "page": "Assembly",
    "title": "JuAFEM.assemble!",
    "category": "function",
    "text": "assemble!(a, Ke, edof)\n\nAssembles the element matrix Ke into a.\n\n\n\nassemble!(g, ge, edof)\n\nAssembles the element residual ge into the global residual vector g.\n\n\n\n"
},

{
    "location": "reference/assembly.html#JuAFEM.end_assemble",
    "page": "Assembly",
    "title": "JuAFEM.end_assemble",
    "category": "function",
    "text": "end_assemble(a::Assembler) -> K\n\nFinalizes an assembly. Returns a sparse matrix with the assembled values.\n\n\n\n"
},

{
    "location": "reference/assembly.html#JuAFEM.create_sparsity_pattern",
    "page": "Assembly",
    "title": "JuAFEM.create_sparsity_pattern",
    "category": "function",
    "text": "create_sparsity_pattern(dh::DofHandler)\n\nCreate the sparsity pattern corresponding to the degree of freedom numbering in the DofHandler. Return a SparseMatrixCSC with stored values in the correct places.\n\nSee the Sparsity Pattern section of the manual.\n\n\n\n"
},

{
    "location": "reference/assembly.html#JuAFEM.create_symmetric_sparsity_pattern",
    "page": "Assembly",
    "title": "JuAFEM.create_symmetric_sparsity_pattern",
    "category": "function",
    "text": "create_symmetric_sparsity_pattern(dh::DofHandler)\n\nCreate the symmetric sparsity pattern corresponding to the degree of freedom numbering in the DofHandler by only considering the upper triangle of the matrix. Return a Symmetric{SparseMatrixCSC}.\n\nSee the Sparsity Pattern section of the manual.\n\n\n\n"
},

{
    "location": "reference/assembly.html#Assembly-1",
    "page": "Assembly",
    "title": "Assembly",
    "category": "section",
    "text": "start_assemble\nassemble!\nend_assemblecreate_sparsity_pattern\ncreate_symmetric_sparsity_pattern"
},

{
    "location": "reference/boundary_conditions.html#",
    "page": "Boundary Conditions",
    "title": "Boundary Conditions",
    "category": "page",
    "text": "DocTestSetup = :(using JuAFEM)"
},

{
    "location": "reference/boundary_conditions.html#JuAFEM.ConstraintHandler",
    "page": "Boundary Conditions",
    "title": "JuAFEM.ConstraintHandler",
    "category": "type",
    "text": "ConstraintHandler\n\nCollection of constraints.\n\n\n\n"
},

{
    "location": "reference/boundary_conditions.html#JuAFEM.Dirichlet",
    "page": "Boundary Conditions",
    "title": "JuAFEM.Dirichlet",
    "category": "type",
    "text": "Dirichlet(u, ∂Ω, f)\nDirichlet(u, ∂Ω, f, component)\n\nCreate a Dirichlet boundary condition on u on the ∂Ω part of the boundary. f is a function that takes two arguments, x and t where x is the spatial coordinate and t is the current time, and returns the prescribed value. For example, here we create a Dirichlet condition for the :u field, on the faceset called ∂Ω and the value given by the sin function:\n\ndbc = Dirichlet(:u, ∂Ω, (x, t) -> sin(t))\n\nIf :u is a vector field we can specify which component the condition should be applied to by specifying component. component can be given either as an integer, or as a vector, for example:\n\ndbc = Dirichlet(:u, ∂Ω, (x, t) -> sin(t), 1)      # applied to component 1\ndbc = Dirichlet(:u, ∂Ω, (x, t) -> sin(t), [1, 3]) # applied to component 1 and 3\n\nDirichlet boundary conditions are added to a ConstraintHandler which applies the condition via apply!.\n\n\n\n"
},

{
    "location": "reference/boundary_conditions.html#JuAFEM.add!",
    "page": "Boundary Conditions",
    "title": "JuAFEM.add!",
    "category": "function",
    "text": "add!(ch::ConstraintHandler, dbc::Dirichlet)\n\nAdd a Dirichlet boundary condition to the ConstraintHandler.\n\n\n\n"
},

{
    "location": "reference/boundary_conditions.html#JuAFEM.close!",
    "page": "Boundary Conditions",
    "title": "JuAFEM.close!",
    "category": "function",
    "text": "close!(ch::ConstraintHandler)\n\nClose and finalize the ConstraintHandler.\n\n\n\n"
},

{
    "location": "reference/boundary_conditions.html#Boundary-Conditions-1",
    "page": "Boundary Conditions",
    "title": "Boundary Conditions",
    "category": "section",
    "text": "ConstraintHandler\nDirichlet\nadd!\nclose!"
},

{
    "location": "reference/grid.html#",
    "page": "Grid",
    "title": "Grid",
    "category": "page",
    "text": "DocTestSetup = :(using JuAFEM)"
},

{
    "location": "reference/grid.html#Grid-1",
    "page": "Grid",
    "title": "Grid",
    "category": "section",
    "text": ""
},

{
    "location": "reference/export.html#",
    "page": "Export",
    "title": "Export",
    "category": "page",
    "text": "DocTestSetup = :(using JuAFEM)"
},

{
    "location": "reference/export.html#WriteVTK.vtk_grid",
    "page": "Export",
    "title": "WriteVTK.vtk_grid",
    "category": "function",
    "text": "vtk_grid(filename::AbstractString, grid::Grid)\n\nCreate a unstructured VTK grid from a Grid. Return a DatasetFile which data can be appended to, see vtk_point_data and vtk_cell_data.\n\n\n\n"
},

{
    "location": "reference/export.html#Export-1",
    "page": "Export",
    "title": "Export",
    "category": "section",
    "text": "vtk_grid"
},

]}
