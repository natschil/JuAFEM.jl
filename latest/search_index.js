var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": "DocTestSetup = quote\n    using JuAFEM\nend"
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
    "text": "JuAFEM is a finite element toolbox that provides functionalities to implement finite element analysis in Julia. The aim is to be general and to keep mathematical abstractions. The main functionalities of the package include:Facilitate integration using different quadrature rules.\nDefine different finite element spaces.\nEvaluate shape functions, derivatives of shape functions etc. for the different finite element spaces and quadrature rules.\nEvaluate functions and derivatives in the finite element space.\nGenerate simple grids.\nExport grids and solutions to VTK.The types and functionalities of JuAFEM is described in more detail in the manual, see below."
},

{
    "location": "index.html#Feature-plans-1",
    "page": "Home",
    "title": "Feature plans",
    "category": "section",
    "text": "JuAFEM is still under heavy development. If you find a bug, or have ideas on how to improve the package, feel free to open an issue or to make a pull request on the JuAFEM GitHub page.The following functionalities are currently in development and will be included in JuAFEM:Import grids from common file formats.\nFacilitate degree of freedom numbering and keeping track of different fields."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "To install, simply run the following in the Julia REPL:Pkg.clone(\"https://github.com/KristofferC/JuAFEM.jl\")and then runusing JuAFEMto load the package."
},

{
    "location": "index.html#API-1",
    "page": "Home",
    "title": "API",
    "category": "section",
    "text": "Pages = [\"lib/maintypes.md\", \"lib/utility_functions.md\"]\nDepth = 2"
},

{
    "location": "man/fe_intro.html#",
    "page": "Introduction to FEM",
    "title": "Introduction to FEM",
    "category": "page",
    "text": ""
},

{
    "location": "man/fe_intro.html#Introduction-to-FEM-1",
    "page": "Introduction to FEM",
    "title": "Introduction to FEM",
    "category": "section",
    "text": "Here, a very brief introduction to the finite element method is given, for reference. As an illustrative example we use balance of momentum, with a linear elastic material."
},

{
    "location": "man/fe_intro.html#Strong-format-1",
    "page": "Introduction to FEM",
    "title": "Strong format",
    "category": "section",
    "text": "The strong format of balance of momentum can be written as:-mathbfsigmacdot mathbfnabla = mathbffwhere mathbfsigma is the stress tensor and mathbff is the internal force (body force). To complete the system of equations we need boundary conditions. These are normally given as known displacements mathbfu^textp or known traction mathbft^textp on different parts of the boundary. The stress can, using the linear elastic material model, be written asmathbfsigma = mathbfmathsfC  mathbfvarepsilon(mathbfu) = mathbfmathsfC  mathbfu otimes mathbfnabla^textsymwhere mathbfmathsfC is the 4th order elasticity tensor and mathbfvarepsilon(mathbfu) the symmetric part of the gradient of the displacement field mathbfu."
},

{
    "location": "man/fe_intro.html#Weak-format-1",
    "page": "Introduction to FEM",
    "title": "Weak format",
    "category": "section",
    "text": "The solution of the equation above is usually calculated from the corresponding weak format. By multiplying the equation with an arbitrary test function mathbfdelta u, integrating over the domain and using partial integration the following equation is obtained; Find mathbfu in mathbbU s.t.int_Omega mathbfvarepsilon(mathbfdelta u)  mathbfmathsfC  mathbfvarepsilon(mathbfu) textdOmega = \nint_Gamma mathbfdelta u cdot mathbft^textp textdGamma +\nint_Omega mathbfdelta u cdot mathbff textdOmega qquad forall mathbfdelta u in mathbbU^0where mathbbU mathbbU^0 are function spaces with sufficiently regular functions. The solution to this equation is identical to the one of the strong format."
},

{
    "location": "man/fe_intro.html#FE-approximation-1",
    "page": "Introduction to FEM",
    "title": "FE-approximation",
    "category": "section",
    "text": "Introduce the finite element approximation mathbfu_h (e.g. mathbfu approx mathbfu_texth) as a sum of shape functions, mathbfN_i and nodal values, a_imathbfu_texth = sum_i=1^textN mathbfN_i a_iqquad deltamathbfu_texth = sum_i=1^textN mathbfN_i delta a_iThis approximation can be inserted in the weak format, which givessum_i^Nsum_j^N delta a_i int_Omega mathbfvarepsilon(mathbfN_i)  mathbfmathsfC  mathbfvarepsilon(mathbfN_j) textdOmega a_j = \nsum_i^N delta a_i int_Gamma mathbfN_i cdot mathbft^textp textdGamma +\nsum_i^N delta a_i int_Omega mathbfN_i cdot mathbff textdOmegaSince mathbfdelta u is arbitrary, the nodal values delta a_i are arbitrary. Thus, the equation can be written asunderlineK underlinea = underlinefwhere underlineK is the stiffness matrix, underlinea is the solution vector with the nodal values and underlinef is the force vector. The elements of underlineK and underlinef are given byunderlineK_ij = int_Omega mathbfvarepsilon(mathbfN_i)  mathbfmathsfC  mathbfvarepsilon(mathbfN_j) textdOmegaunderlinef_i = int_Gamma mathbfN_i cdot mathbft^textp textdGamma +\n                     int_Omega mathbfN_i cdot mathbff textdOmegaThe solution to the system (which in this case is linear) is simply given by inverting the matrix underlineK and using the boundary conditions (prescribed displacements)underlinea = underlineK^-1 underlinef"
},

{
    "location": "man/fe_intro.html#Implementation-in-JuAFEM-1",
    "page": "Introduction to FEM",
    "title": "Implementation in JuAFEM",
    "category": "section",
    "text": "In practice, the shape functions mathbfN are only non-zero on parts of the domain Omega. Thus, the integrals are evaluated on sub-domains, called elements or cells. All the cells gives a contribution to the global stiffness matrix and force vector.The integrals are evaluated using quadrature. In JuAFEM the stiffness matrix and force vector can be calculated like this...\n\nfor qp in 1:Nqp\n    for i in 1:N\n        f[i] += shape_value(i) ⋅ f * dΩ\n        for j in 1:N\n            K[i,j] += shape_symmetric_gradient(i) : C : shape_symmetric_gradient(j) * dΩ\n        end\n    end\nend\n\n...Although this is a simplification of the actual code, note the similarity between the code and the mathematical expression above."
},

{
    "location": "man/getting_started.html#",
    "page": "Getting started",
    "title": "Getting started",
    "category": "page",
    "text": "DocTestSetup = quote\n    using JuAFEM\n    quad_rule = QuadratureRule{2, RefCube}(2)\n    func_space = Lagrange{2, RefCube, 1}()\n    cell_values = CellScalarValues(quad_rule, func_space)\n    x = Vec{2, Float64}[Vec{2}((0.0, 0.0)),\n                           Vec{2}((1.5, 0.0)),\n                           Vec{2}((2.0, 2.0)),\n                           Vec{2}((0.0, 1.0))]\n    reinit!(cell_values, x)\nend"
},

{
    "location": "man/getting_started.html#Getting-started-1",
    "page": "Getting started",
    "title": "Getting started",
    "category": "section",
    "text": "tip: Tip\nCheckout some examples of usage of JuAFEM in the examples/ directory.For the impatient: Here is a quick overview on how the some of the packages functionalities can be used. This quickly describes CellScalarValues which a lot of the package is built upon.First, create a quadrature rule, for integration in 2D, on a reference cube:julia> quad_rule = QuadratureRule{2, RefCube}(2);Next, create a function spacejulia> func_space = Lagrange{2, RefCube, 1}();Use these to create a CellScalarValues object.julia> cell_values = CellScalarValues(quad_rule, func_space);Presume one cell in the grid has the following vertices:julia> x = Vec{2, Float64}[Vec{2}((0.0, 0.0)),\n                           Vec{2}((1.5, 0.0)),\n                           Vec{2}((2.0, 2.0)),\n                           Vec{2}((0.0, 1.0))];To update cell_values for the given cell, use reinit!:julia> reinit!(cell_values, x)We can now query the CellScalarValues object for shape function information:Value of shape function 1 in quadrature point 3julia> shape_value(cell_values, 3, 1)\n0.16666666666666663Derivative of the same shape function, in the same quadrature pointjulia> shape_gradient(cell_values, 3, 1)\n2-element ContMechTensors.Tensor{1,2,Float64,2}:\n  0.165523\n -0.665523We can also evaluate values and gradients of functions on the finite element basis.julia> T = [0.0, 1.0, 2.0, 1.5]; # nodal values\n\njulia> function_value(cell_values, 3, T) # value of T in 3rd quad point\n1.311004233964073\n\njulia> function_gradient(cell_values, 1, T)  # value of grad(T) in 1st quad point\n2-element ContMechTensors.Tensor{1,2,Float64,2}:\n 0.410202\n 1.1153The same can also be done for a vector valued function:julia> u = Vec{2, Float64}[Vec{2}((0.0, 0.0)),\n                           Vec{2}((3.5, 2.0)),\n                           Vec{2}((2.0, 2.0)),\n                           Vec{2}((2.0, 1.0))]; # nodal vectors\n\njulia> function_value(cell_values, 2, u) # value of u in 2nd quad point\n2-element ContMechTensors.Tensor{1,2,Float64,2}:\n 2.59968\n 1.62201\n\njulia> function_symmetric_gradient(cell_values, 3, u) # sym(grad(u)) in 3rd quad point\n2×2 ContMechTensors.SymmetricTensor{2,2,Float64,3}:\n -0.0443518  0.713306\n  0.713306   0.617741For more functions see the documentation for CellValues"
},

{
    "location": "lib/maintypes.html#",
    "page": "Main Types",
    "title": "Main Types",
    "category": "page",
    "text": "CurrentModule = JuAFEM\nDocTestSetup = quote\n    using JuAFEM\nend"
},

{
    "location": "lib/maintypes.html#JuAFEM.AbstractRefShape",
    "page": "Main Types",
    "title": "JuAFEM.AbstractRefShape",
    "category": "Type",
    "text": "Represents a reference shape which quadrature rules and function spaces are defined on. Currently, the only concrete types that subtype this type are RefCube in 1,2 and 3 dimensions, and RefTetrahedron in 2 and 3 dimensions.\n\n\n\n"
},

{
    "location": "lib/maintypes.html#JuAFEM.QuadratureRule",
    "page": "Main Types",
    "title": "JuAFEM.QuadratureRule",
    "category": "Type",
    "text": "A QuadratureRule is used to approximate an integral on a domain by a weighted sum of function values at specific points:\n\nintlimits_Omega f(mathbfx) textd Omega approx sumlimits_q = 1^n_q f(mathbfx_q) w_q\n\nThe quadrature rule consists of n_q points in space mathbfx_q with corresponding weights w_q.\n\nThere are different rules to determine the points and weights. In JuAFEM two different types are implemented: :legendre and :lobatto, where :lobatto is only supported for RefCube. If the quadrature rule type is left out, :legendre is used by default.\n\nIn JuAFEM, the QuadratureRule type is mostly used as one of the components to create a CellValues or BoundaryValues object.\n\nConstructor:\n\nQuadratureRule{dim, shape}([quad_rule_type::Symbol], order::Int)\n\nArguments:\n\ndim: the space dimension of the reference shape\nshape: an AbstractRefShape\nquad_rule_type: :legendre or :lobatto, defaults to :legendre.\norder: the order of the quadrature rule\n\nCommon methods:\n\ngetpoints : the points of the quadrature rule\ngetweights : the weights of the quadrature rule\n\nExample:\n\njulia> QuadratureRule{2, RefTetrahedron}(1)\nJuAFEM.QuadratureRule{2,JuAFEM.RefTetrahedron,Float64}([0.5],ContMechTensors.Tensor{1,2,Float64,2}[[0.333333,0.333333]])\n\njulia> QuadratureRule{1, RefCube}(:lobatto, 2)\nJuAFEM.QuadratureRule{1,JuAFEM.RefCube,Float64}([1.0,1.0],ContMechTensors.Tensor{1,1,Float64,1}[[-1.0],[1.0]])\n\n\n\n"
},

{
    "location": "lib/maintypes.html#JuAFEM.FunctionSpace",
    "page": "Main Types",
    "title": "JuAFEM.FunctionSpace",
    "category": "Type",
    "text": "A FunctionSpace is used to define shape functions.\n\nConstructor:\n\nFunctionSpace{dim, reference_shape, order}()\n\nArguments:\n\ndim: the dimension the function space lives in\nshape: a reference shape, see AbstractRefShape\norder: the highest order term in the polynomial\n\nThe following function spaces are implemented:\n\nLagrange{1, RefCube, 1}\nLagrange{1, RefCube, 2}\nLagrange{2, RefCube, 1}\nLagrange{2, RefCube, 2}\nLagrange{2, RefTetrahedron, 1}\nLagrange{2, RefTetrahedron, 2}\nLagrange{3, RefCube, 1}\nSerendipity{2, RefCube, 2}\nLagrange{3, RefTetrahedron, 1}\n\nCommon methods:\n\ngetnbasefunctions\ngetdim\ngetrefshape\ngetorder\n\nExample:\n\njulia> fs = Lagrange{2, RefTetrahedron, 2}()\nJuAFEM.Lagrange{2,JuAFEM.RefTetrahedron,2}()\n\njulia> getnbasefunctions(fs)\n6\n\n\n\n"
},

{
    "location": "lib/maintypes.html#JuAFEM.CellValues",
    "page": "Main Types",
    "title": "JuAFEM.CellValues",
    "category": "Type",
    "text": "A CellValues object facilitates the process of evaluating values shape functions, gradients of shape functions, values of nodal functions, gradients and divergences of nodal functions etc. in the finite element cell. There are two different types of CellValues: CellScalarValues and CellVectorValues. As the names suggest, CellScalarValues utilizes scalar shape functions and CellVectorValues utilizes vectorial shape functions. For a scalar field, the CellScalarValues type should be used. For vector field, both subtypes can be used.\n\nConstructors:\n\nCellScalarValues([::Type{T}], quad_rule::QuadratureRule, function_space::FunctionSpace, [geometric_space::FunctionSpace])\nCellVectorValues([::Type{T}], quad_rule::QuadratureRule, function_space::FunctionSpace, [geometric_space::FunctionSpace])\n\nArguments:\n\nT: an optional argument to determine the type the internal data is stored as.\nquad_rule: an instance of a QuadratureRule\nfunction_space: an instance of a FunctionSpace used to interpolate the approximated function\ngeometric_space: an optional instance of a FunctionSpace which is used to interpolate the geometry\n\nCommon methods:\n\nreinit!\ngetnquadpoints\ngetquadrule\ngetfunctionspace\ngetgeometricspace\ngetdetJdV\nshape_value\nshape_gradient\nshape_symmetric_gradient\nshape_divergence\nfunction_value\nfunction_gradient\nfunction_symmetric_gradient\nfunction_divergence\nspatial_coordinate\n\n\n\n"
},

{
    "location": "lib/maintypes.html#JuAFEM.BoundaryValues",
    "page": "Main Types",
    "title": "JuAFEM.BoundaryValues",
    "category": "Type",
    "text": "A BoundaryValues object facilitates the process of evaluating values shape functions, gradients of shape functions, values of nodal functions, gradients and divergences of nodal functions etc. on the finite element boundary. There are two different types of BoundaryValues: BoundaryScalarValues and BoundaryVectorValues. As the names suggest, BoundaryScalarValues utilizes scalar shape functions and BoundaryVectorValues utilizes vectorial shape functions. For a scalar field, the BoundaryScalarValues type should be used. For vector field, both subtypes can be used.\n\nConstructors:\n\nNote: The quadrature rule for the boundary should be given with one dimension lower. I.e. for a 3D case, the quadrature rule should be in 2D.\n\nBoundaryScalarValues([::Type{T}], quad_rule::QuadratureRule, function_space::FunctionSpace, [geometric_space::FunctionSpace])\nBoundaryVectorValues([::Type{T}], quad_rule::QuadratureRule, function_space::FunctionSpace, [geometric_space::FunctionSpace])\n\nArguments:\n\nT: an optional argument to determine the type the internal data is stored as.\nquad_rule: an instance of a QuadratureRule\nfunction_space: an instance of a FunctionSpace used to interpolate the approximated function\ngeometric_space: an optional instance of a FunctionSpace which is used to interpolate the geometry\n\nCommon methods:\n\nreinit!\ngetboundarynumber\ngetnquadpoints\ngetquadrule\ngetfunctionspace\ngetgeometricspace\ngetdetJdV\nshape_value\nshape_gradient\nshape_symmetric_gradient\nshape_divergence\nfunction_value\nfunction_gradient\nfunction_symmetric_gradient\nfunction_divergence\nspatial_coordinate\n\n\n\n"
},

{
    "location": "lib/maintypes.html#Main-Types-1",
    "page": "Main Types",
    "title": "Main Types",
    "category": "section",
    "text": "Pages = [\"maintypes.md\"]AbstractRefShape\nQuadratureRule\nFunctionSpace\nCellValues\nBoundaryValues"
},

{
    "location": "lib/utility_functions.html#",
    "page": "Utilities",
    "title": "Utilities",
    "category": "page",
    "text": "CurrentModule = JuAFEM\nDocTestSetup = quote\n    using JuAFEM\nend"
},

{
    "location": "lib/utility_functions.html#Utilities-1",
    "page": "Utilities",
    "title": "Utilities",
    "category": "section",
    "text": "Pages = [\"utility_functions.md\"]"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getpoints",
    "page": "Utilities",
    "title": "JuAFEM.getpoints",
    "category": "Function",
    "text": "The points of the quadrature rule.\n\ngetpoints(qr::QuadratureRule)\n\nArguments:\n\nqr: the quadrature rule\n\nExample:\n\njulia> qr = QuadratureRule{2, RefTetrahedron}(:legendre, 2);\n\njulia> getpoints(qr)\n3-element Array{ContMechTensors.Tensor{1,2,Float64,2},1}:\n [0.166667,0.166667]\n [0.166667,0.666667]\n [0.666667,0.166667]\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getweights",
    "page": "Utilities",
    "title": "JuAFEM.getweights",
    "category": "Function",
    "text": "The weights of the quadrature rule.\n\ngetweights(qr::QuadratureRule) = qr.weights\n\nArguments:\n\nqr: the quadrature rule\n\nExample:\n\njulia> qr = QuadratureRule{2, RefTetrahedron}(:legendre, 2);\n\njulia> getweights(qr)\n3-element Array{Float64,1}:\n 0.166667\n 0.166667\n 0.166667\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#QuadratureRule-1",
    "page": "Utilities",
    "title": "QuadratureRule",
    "category": "section",
    "text": "getpoints\ngetweights"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getnbasefunctions",
    "page": "Utilities",
    "title": "JuAFEM.getnbasefunctions",
    "category": "Function",
    "text": "Returns the number of base functions for a FunctionSpace or Values object.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getdim",
    "page": "Utilities",
    "title": "JuAFEM.getdim",
    "category": "Function",
    "text": "Returns the dimension of a FunctionSpace\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getrefshape",
    "page": "Utilities",
    "title": "JuAFEM.getrefshape",
    "category": "Function",
    "text": "Returns the reference shape of a FunctionSpace\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getorder",
    "page": "Utilities",
    "title": "JuAFEM.getorder",
    "category": "Function",
    "text": "Returns the polynomial order of the FunctionSpace\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#FunctionSpace-1",
    "page": "Utilities",
    "title": "FunctionSpace",
    "category": "section",
    "text": "getnbasefunctions\ngetdim\ngetrefshape\ngetorder"
},

{
    "location": "lib/utility_functions.html#JuAFEM.reinit!",
    "page": "Utilities",
    "title": "JuAFEM.reinit!",
    "category": "Function",
    "text": "Updates a CellValues/BoundaryValues object for a cell or boundary.\n\nreinit!{dim, T}(cv::CellValues{dim}, x::Vector{Vec{dim, T}})\nreinit!{dim, T}(bv::BoundaryValues{dim}, x::Vector{Vec{dim, T}}, boundary::Int)\n\nArguments:\n\ncv/bv: the CellValues/BoundaryValues object\nx: a Vector of Vec, one for each nodal position in the element.\nboundary: an integer to specify which boundary of the cell\n\nResult\n\nnothing\n\nDetails\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getnquadpoints",
    "page": "Utilities",
    "title": "JuAFEM.getnquadpoints",
    "category": "Function",
    "text": "The number of quadrature points for  the Values type.\n\ngetnquadpoints(fe_v::Values)\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getquadrule",
    "page": "Utilities",
    "title": "JuAFEM.getquadrule",
    "category": "Function",
    "text": "The quadrature rule for the Values type.\n\ngetquadrule(fe_v::Values)\n\n** Arguments **\n\nfe_v: the Values object\n\n** Results **\n\n::QuadratureRule: the quadrature rule.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getfunctionspace",
    "page": "Utilities",
    "title": "JuAFEM.getfunctionspace",
    "category": "Function",
    "text": "The function space for the Values type.\n\ngetfunctionspace(fe_v::Values)\n\nArguments\n\nfe_v: the Values object\n\nResults\n\n::FunctionSpace: the function space\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getgeometricspace",
    "page": "Utilities",
    "title": "JuAFEM.getgeometricspace",
    "category": "Function",
    "text": "The function space used for geometric interpolation for the Values type.\n\ngetgeometricspace(fe_v::Values)\n\nArguments\n\nfe_v: the Values object\n\nResults\n\n::FunctionSpace: the geometric interpolation function space\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getdetJdV",
    "page": "Utilities",
    "title": "JuAFEM.getdetJdV",
    "category": "Function",
    "text": "The product between the determinant of the Jacobian and the quadrature point weight for a given quadrature point: det(J(mathbfx)) w_q\n\ngetdetJdV(fe_v::Values, quadrature_point::Int)\n\n** Arguments:**\n\nfe_v: the Values object\nquadrature_point The quadrature point number\n\nResults:\n\n::Number\n\nDetails:\n\nThis value is typically used when one integrates a function on a finite element cell or boundary as\n\nintlimits_Omega f(mathbfx) d Omega approx sumlimits_q = 1^n_q f(mathbfx_q) det(J(mathbfx)) w_q intlimits_Gamma f(mathbfx) d Gamma approx sumlimits_q = 1^n_q f(mathbfx_q) det(J(mathbfx)) w_q\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.shape_value",
    "page": "Utilities",
    "title": "JuAFEM.shape_value",
    "category": "Function",
    "text": "Computes the value of the shape function\n\nshape_value(fe_v::Values, quadrature_point::Int, base_function::Int)\n\nGets the values of the shape function for a given quadrature point and base_func\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.shape_gradient",
    "page": "Utilities",
    "title": "JuAFEM.shape_gradient",
    "category": "Function",
    "text": "Get the gradient of the shape functions for a given quadrature point and base function\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.shape_symmetric_gradient",
    "page": "Utilities",
    "title": "JuAFEM.shape_symmetric_gradient",
    "category": "Function",
    "text": "Get the symmetric gradient of the shape functions for a given quadrature point and base function\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.shape_divergence",
    "page": "Utilities",
    "title": "JuAFEM.shape_divergence",
    "category": "Function",
    "text": "Get the divergence of the shape functions for a given quadrature point and base function\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.function_value",
    "page": "Utilities",
    "title": "JuAFEM.function_value",
    "category": "Function",
    "text": "Computes the value in a quadrature point for a scalar or vector valued function\n\nfunction_value{dim, T}(fe_v::Values{dim}, q_point::Int, u::Vector{T})\nfunction_value{dim, T}(fe_v::Values{dim}, q_point::Int, u::Vector{Vec{dim, T}})\n\nArguments:\n\nfe_v: the Values object\nq_point: the quadrature point number\nu: the value of the function in the nodes\n\nResults:\n\n::Number: the value of a scalar valued function\n::Vec{dim, T} the value of a vector valued function\n\nDetails:\n\nThe value of a scalar valued function is computed as u(mathbfx) = sumlimits_i = 1^n N_i (mathbfx) u_i where u_i are the value of u in the nodes. For a vector valued function the value is calculated as mathbfu(mathbfx) = sumlimits_i = 1^n N_i (mathbfx) mathbfu_i where mathbfu_i are the nodal values of mathbfu.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.function_gradient",
    "page": "Utilities",
    "title": "JuAFEM.function_gradient",
    "category": "Function",
    "text": "Computes the gradient in a quadrature point for a scalar or vector valued function\n\nfunction_scalar_gradient{dim, T}(fe_v::Values{dim}, q_point::Int, u::Vector{T})\nfunction_vector_gradient{dim, T}(fe_v::Values{dim}, q_point::Int, u::Vector{Vec{dim, T}})\n\nArguments:\n\nfe_v: the Values object\nq_point: the quadrature point number\nu: the value of the function in the nodes\n\nResults:\n\n::Vec{dim, T}: the gradient of a scalar valued function\n::Tensor{2, dim, T}: the gradient of a vector valued function\n\nDetails:\n\nThe gradient of a scalar function is computed as mathbfnabla u(mathbfx) = sumlimits_i = 1^n mathbfnabla N_i (mathbfx) u_i where u_i are the nodal values of the function. For a vector valued function the gradient is computed as mathbfnabla mathbfu(mathbfx) = sumlimits_i = 1^n mathbfnabla N_i (mathbfx) otimes mathbfu_i where mathbfu_i are the nodal values of mathbfu.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.function_symmetric_gradient",
    "page": "Utilities",
    "title": "JuAFEM.function_symmetric_gradient",
    "category": "Function",
    "text": "Computes the symmetric gradient for a vector valued function in a quadrature point.\n\nfunction_symmetric_gradient{dim, T}(fe_v::Values{dim}, q_point::Int, u::Vector{Vec{dim, T}})\n\nArguments:\n\nfe_v: the Values object\nq_point: the quadrature point number\nu: the value of the function in the nodes\n\nResults:\n\n::SymmetricTensor{2, dim, T}: the symmetric gradient\n\nDetails:\n\nThe symmetric gradient of a scalar function is computed as\n\nleft mathbfnabla  mathbfu(mathbfx_q) right^textsym =  sumlimits_i = 1^n  frac12 left mathbfnabla N_i (mathbfx_q) otimes mathbfu_i + mathbfu_i  otimes  mathbfnabla N_i (mathbfx_q) right\n\nwhere mathbfu_i are the nodal values of the function.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.function_divergence",
    "page": "Utilities",
    "title": "JuAFEM.function_divergence",
    "category": "Function",
    "text": "Computes the divergence in a quadrature point for a vector valued function.\n\nfunction_divergence{dim, T}(fe_v::Values{dim}, q_point::Int, u::Vector{Vec{dim, T}})\n\nArguments:\n\nfe_v: the Values object\nq_point: the quadrature point number\nu: the value of the function in the nodes\n\nResults:\n\n::Number: the divergence of the function\n\nDetails:\n\nThe divergence of a vector valued functions in the quadrature point mathbfx_q) is computed as\n\nmathbfnabla cdot mathbfu(mathbfx_q) = sumlimits_i = 1^n mathbfnabla N_i (mathbfx_q) cdot mathbfu_i\n\nwhere mathbfu_i are the nodal values of the function.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.spatial_coordinate",
    "page": "Utilities",
    "title": "JuAFEM.spatial_coordinate",
    "category": "Function",
    "text": "spatial_coordinate{dim, T}(fe_v::Values{dim}, q_point::Int, x::Vector{Vec{dim, T}})\n\nComputes the spatial coordinate in a quadrature point.\n\nArguments:\n\nfe_v: the Values object\nq_point: the quadrature point number\nx: the nodal coordinates of the cell\n\nResults:\n\n::Vec{dim, T}: the spatial coordinate\n\nDetails:\n\nThe coordinate is computed, using the geometric interpolation space, as mathbfx = sumlimits_i = 1^n M_i (mathbfx) mathbfhatx_i\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#CellValues-1",
    "page": "Utilities",
    "title": "CellValues",
    "category": "section",
    "text": "reinit!\ngetnquadpoints\ngetquadrule\ngetfunctionspace\ngetgeometricspace\ngetdetJdV\n\nshape_value\nshape_gradient\nshape_symmetric_gradient\nshape_divergence\n\nfunction_value\nfunction_gradient\nfunction_symmetric_gradient\nfunction_divergence\nspatial_coordinate"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getboundarynumber",
    "page": "Utilities",
    "title": "JuAFEM.getboundarynumber",
    "category": "Function",
    "text": "The boundary number for a cell, typically used to get the boundary number which is needed to reinit! a BoundaryValues object for  boundary integration\n\ngetboundarynumber(boundary_nodes, cell_nodes, fs::FunctionSpace)\n\n** Arguments **\n\nboundary_nodes: the node numbers of the nodes on the boundary of the cell\ncell_nodes: the node numbers of the cell\nfs: the FunctionSpace for the cell\n\n** Results **\n\n::Int: the corresponding boundary\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getcurrentboundary",
    "page": "Utilities",
    "title": "JuAFEM.getcurrentboundary",
    "category": "Function",
    "text": "The current active boundary of the BoundaryValues type.\n\ngetcurrentboundary(bv::BoundaryScalarValues)\n\n** Arguments **\n\nbv: the BoundaryValues object\n\n** Results **\n\n::Int: the current active boundary (from last reinit!).\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#BoundaryValues-1",
    "page": "Utilities",
    "title": "BoundaryValues",
    "category": "section",
    "text": "All of the methods for CellValues apply for BoundaryValues as well. In addition, there are some methods that are unique for BoundaryValues:getboundarynumber\ngetcurrentboundary"
},

{
    "location": "lib/utility_functions.html#WriteVTK.vtk_grid",
    "page": "Utilities",
    "title": "WriteVTK.vtk_grid",
    "category": "Function",
    "text": "Creates an unstructured VTK grid from the element topology and coordinates.\n\nvtk_grid{dim,T}(filename::AbstractString, coords::Vector{Vec{dim,T}}, topology::Matrix{Int}, celltype::VTKCellTypes.VTKCellType)\n\nArguments\n\nfilename: name (or path) of the file when it is saved to disk, eg filename = \"myfile\", or filename = \"/results/myfile\" to store it in the folder results\ncoords: a vector of the node coordinates\ntopology: a matrix where each column contains the nodes which connects the element\ncelltype: the definition of the celltype in the grid, see https://github.com/jipolanco/WriteVTK.jl#defining-cells\n\nResults:\n\n::DatasetFile\n\nExample:\n\njulia> coords = [Vec{2}((0.0,0.0)), Vec{2}((1.0,0.0)), Vec{2}((1.5,1.5)), Vec{2}((0.0,1.0))]\n4-element Array{ContMechTensors.Tensor{1,2,Float64,2},1}:\n [0.0,0.0]\n [1.0,0.0]\n [1.5,1.5]\n [0.0,1.0]\n\njulia> topology = [1 2 4; 2 3 4]'\n3×2 Array{Int64,2}:\n 1  2\n 2  3\n 4  4\n\njulia> celltype = VTKCellTypes.VTK_TRIANGLE;\n\njulia> vtkobj = vtk_grid(\"example\", coords, topology, celltype);\n\njulia> vtk_save(vtkobj)\n1-element Array{String,1}:\n \"example.vtu\"\n\nDetails\n\nThis is a thin wrapper around the vtk_grid function from the WriteVTK package.\n\nFor information how to add cell data and point data to the resulting VTK object as well as how to write it to a file see https://github.com/jipolanco/WriteVTK.jl#generating-an-unstructured-vtk-file\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getVTKtype",
    "page": "Utilities",
    "title": "JuAFEM.getVTKtype",
    "category": "Function",
    "text": "Returns the VTKCellType corresponding to the input FunctionSpace\n\ngetVTKtype(fs::FunctionSpace)\n\nArguments\n\nfs: The function space\n\nResults:\n\n::VTKCellType: The cell type, see https://github.com/jipolanco/WriteVTK.jl#generating-an-unstructured-vtk-file\n\nExample:\n\njulia> fs = Lagrange{2, RefCube, 1}()\nJuAFEM.Lagrange{2,JuAFEM.RefCube,1}()\n\njulia> getVTKtype(fs)\nWriteVTK.VTKCellTypes.VTKCellType(\"VTK_QUAD\",0x09,4)\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#VTK-1",
    "page": "Utilities",
    "title": "VTK",
    "category": "section",
    "text": "vtk_grid\ngetVTKtype"
},

]}
