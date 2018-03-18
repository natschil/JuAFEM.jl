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
    "text": "JuAFEM is a finite element toolbox that provides functionalities to implement finite element analysis in Julia. The aim is to be general and to keep mathematical abstractions. The main functionalities of the package include:Facilitate integration using different quadrature rules.\nDefine different finite element interpolations.\nEvaluate shape functions, derivatives of shape functions etc. for the different interpolations and quadrature rules.\nEvaluate functions and derivatives in the finite element space.\nGenerate simple grids.\nExport grids and solutions to VTK.The types and functionalities of JuAFEM is described in more detail in the manual, see below."
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
    "text": "In practice, the shape functions mathbfN are only non-zero on parts of the domain Omega. Thus, the integrals are evaluated on sub-domains, called elements or cells. All the cells gives a contribution to the global stiffness matrix and force vector.The integrals are evaluated using quadrature. In JuAFEM the stiffness matrix and force vector can be calculated like this...\n\nfor qp in 1:Nqp\n    for i in 1:N\n        f[i] += (shape_value(i) ⋅ f) * dΩ\n        for j in 1:N\n            K[i,j] += (shape_symmetric_gradient(i) : C : shape_symmetric_gradient(j)) * dΩ\n        end\n    end\nend\n\n...Although this is a simplification of the actual code, note the similarity between the code and the mathematical expression above."
},

{
    "location": "man/getting_started.html#",
    "page": "Getting started",
    "title": "Getting started",
    "category": "page",
    "text": "DocTestSetup = quote\n    using JuAFEM\n    quad_rule = QuadratureRule{2, RefCube}(2)\n    interpolation = Lagrange{2, RefCube, 1}()\n    cell_values = CellScalarValues(quad_rule, interpolation)\n    x = Vec{2, Float64}[Vec{2}((0.0, 0.0)),\n                           Vec{2}((1.5, 0.0)),\n                           Vec{2}((2.0, 2.0)),\n                           Vec{2}((0.0, 1.0))]\n    reinit!(cell_values, x)\nend"
},

{
    "location": "man/getting_started.html#Getting-started-1",
    "page": "Getting started",
    "title": "Getting started",
    "category": "section",
    "text": "tip: Tip\nCheckout some examples of usage of JuAFEM in the examples/ directory.For the impatient: Here is a quick overview on how the some of the packages functionalities can be used. This quickly describes CellScalarValues which a lot of the package is built upon.First, create a quadrature rule, for integration in 2D, on a reference cube:julia> quad_rule = QuadratureRule{2, RefCube}(2);Next, create an interpolationjulia> interpolation = Lagrange{2, RefCube, 1}();Use these to create a CellScalarValues object.julia> cell_values = CellScalarValues(quad_rule, interpolation);Presume one cell in the grid has the following vertices:julia> x = Vec{2, Float64}[Vec{2}((0.0, 0.0)),\n                           Vec{2}((1.5, 0.0)),\n                           Vec{2}((2.0, 2.0)),\n                           Vec{2}((0.0, 1.0))];To update cell_values for the given cell, use reinit!:julia> reinit!(cell_values, x)We can now query the CellScalarValues object for shape function information:Value of shape function 1 in quadrature point 3julia> shape_value(cell_values, 3, 1)\n0.16666666666666669Derivative of the same shape function, in the same quadrature pointjulia> shape_gradient(cell_values, 3, 1)\n2-element Tensors.Tensor{1,2,Float64,2}:\n  0.165523\n -0.665523We can also evaluate values and gradients of functions on the finite element basis.julia> T = [0.0, 1.0, 2.0, 1.5]; # nodal values\n\njulia> function_value(cell_values, 3, T) # value of T in 3rd quad point\n1.3110042339640733\n\njulia> function_gradient(cell_values, 1, T)  # value of grad(T) in 1st quad point\n2-element Tensors.Tensor{1,2,Float64,2}:\n 0.410202\n 1.1153The same can also be done for a vector valued function:julia> u = Vec{2, Float64}[Vec{2}((0.0, 0.0)),\n                           Vec{2}((3.5, 2.0)),\n                           Vec{2}((2.0, 2.0)),\n                           Vec{2}((2.0, 1.0))]; # nodal vectors\n\njulia> function_value(cell_values, 2, u) # value of u in 2nd quad point\n2-element Tensors.Tensor{1,2,Float64,2}:\n 2.59968\n 1.62201\n\njulia> function_symmetric_gradient(cell_values, 3, u) # sym(grad(u)) in 3rd quad point\n2×2 Tensors.SymmetricTensor{2,2,Float64,3}:\n -0.0443518  0.713306\n  0.713306   0.617741For more functions see the documentation for CellValues."
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
    "category": "type",
    "text": "Represents a reference shape which quadrature rules and interpolations are defined on. Currently, the only concrete types that subtype this type are RefCube in 1, 2 and 3 dimensions, and RefTetrahedron in 2 and 3 dimensions.\n\n\n\n"
},

{
    "location": "lib/maintypes.html#JuAFEM.QuadratureRule",
    "page": "Main Types",
    "title": "JuAFEM.QuadratureRule",
    "category": "type",
    "text": "QuadratureRule{dim,shape}([quad_rule_type::Symbol], order::Int)\n\nCreate a QuadratureRule used for integration. dim is the space dimension, shape an AbstractRefShape and order the order of the quadrature rule. quad_rule_type is an optional argument determining the type of quadrature rule, currently the :legendre and :lobatto rules are implemented.\n\nA QuadratureRule is used to approximate an integral on a domain by a weighted sum of function values at specific points:\n\nintlimits_Omega f(mathbfx) textd Omega approx sumlimits_q = 1^n_q f(mathbfx_q) w_q\n\nThe quadrature rule consists of n_q points in space mathbfx_q with corresponding weights w_q.\n\nIn JuAFEM, the QuadratureRule type is mostly used as one of the components to create a CellValues or FaceValues object.\n\nCommon methods:\n\ngetpoints : the points of the quadrature rule\ngetweights : the weights of the quadrature rule\n\nExample:\n\njulia> QuadratureRule{2, RefTetrahedron}(1)\nJuAFEM.QuadratureRule{2,JuAFEM.RefTetrahedron,Float64}([0.5], Tensors.Tensor{1,2,Float64,2}[[0.333333, 0.333333]])\n\njulia> QuadratureRule{1, RefCube}(:lobatto, 2)\nJuAFEM.QuadratureRule{1,JuAFEM.RefCube,Float64}([1.0, 1.0], Tensors.Tensor{1,1,Float64,1}[[-1.0], [1.0]])\n\n\n\n"
},

{
    "location": "lib/maintypes.html#JuAFEM.Interpolation",
    "page": "Main Types",
    "title": "JuAFEM.Interpolation",
    "category": "type",
    "text": "Interpolation{dim, ref_shape, order}()\n\nReturn an Interpolation of given dimension dim, reference shape (see see AbstractRefShape) ref_shape and order order. order corresponds to the highest order term in the polynomial. The interpolation is used to define shape functions to interpolate a function between nodes.\n\nThe following interpolations are implemented:\n\nLagrange{1,RefCube,1}\nLagrange{1,RefCube,2}\nLagrange{2,RefCube,1}\nLagrange{2,RefCube,2}\nLagrange{2,RefTetrahedron,1}\nLagrange{2,RefTetrahedron,2}\nLagrange{3,RefCube,1}\nSerendipity{2,RefCube,2}\nLagrange{3,RefTetrahedron,1}\nLagrange{3,RefTetrahedron,2}\n\nCommon methods:\n\ngetnbasefunctions\ngetdim\ngetrefshape\ngetorder\n\nExamples\n\njulia> ip = Lagrange{2,RefTetrahedron,2}()\nJuAFEM.Lagrange{2,JuAFEM.RefTetrahedron,2}()\n\njulia> getnbasefunctions(ip)\n6\n\n\n\n"
},

{
    "location": "lib/maintypes.html#JuAFEM.CellValues",
    "page": "Main Types",
    "title": "JuAFEM.CellValues",
    "category": "type",
    "text": "CellScalarValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])\nCellVectorValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])\n\nA CellValues object facilitates the process of evaluating values of shape functions, gradients of shape functions, values of nodal functions, gradients and divergences of nodal functions etc. in the finite element cell. There are two different types of CellValues: CellScalarValues and CellVectorValues. As the names suggest, CellScalarValues utilizes scalar shape functions and CellVectorValues utilizes vectorial shape functions. For a scalar field, the CellScalarValues type should be used. For vector field, both subtypes can be used.\n\nArguments:\n\nT: an optional argument (default to Float64) to determine the type the internal data is stored as.\nquad_rule: an instance of a QuadratureRule\nfunc_interpol: an instance of an Interpolation used to interpolate the approximated function\ngeom_interpol: an optional instance of a Interpolation which is used to interpolate the geometry\n\nCommon methods:\n\nreinit!\ngetnquadpoints\ngetdetJdV\nshape_value\nshape_gradient\nshape_symmetric_gradient\nshape_divergence\nfunction_value\nfunction_gradient\nfunction_symmetric_gradient\nfunction_divergence\nspatial_coordinate\n\n\n\n"
},

{
    "location": "lib/maintypes.html#JuAFEM.FaceValues",
    "page": "Main Types",
    "title": "JuAFEM.FaceValues",
    "category": "type",
    "text": "FaceScalarValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])\nFaceVectorValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])\n\nA FaceValues object facilitates the process of evaluating values of shape functions, gradients of shape functions, values of nodal functions, gradients and divergences of nodal functions etc. on the faces of finite elements. There are two different types of FaceValues: FaceScalarValues and FaceVectorValues. As the names suggest, FaceScalarValues utilizes scalar shape functions and FaceVectorValues utilizes vectorial shape functions. For a scalar field, the FaceScalarValues type should be used. For vector field, both subtypes can be used.\n\nnote: Note\nThe quadrature rule for the face should be given with one dimension lower. I.e. for a 3D case, the quadrature rule should be in 2D.\n\nArguments:\n\nT: an optional argument to determine the type the internal data is stored as.\nquad_rule: an instance of a QuadratureRule\nfunc_interpol: an instance of an Interpolation used to interpolate the approximated function\ngeom_interpol: an optional instance of an Interpolation which is used to interpolate the geometry\n\nCommon methods:\n\nreinit!\ngetnquadpoints\ngetdetJdV\nshape_value\nshape_gradient\nshape_symmetric_gradient\nshape_divergence\nfunction_value\nfunction_gradient\nfunction_symmetric_gradient\nfunction_divergence\nspatial_coordinate\n\n\n\n"
},

{
    "location": "lib/maintypes.html#Main-Types-1",
    "page": "Main Types",
    "title": "Main Types",
    "category": "section",
    "text": "Pages = [\"maintypes.md\"]AbstractRefShape\nQuadratureRule\nInterpolation\nCellValues\nFaceValues"
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
    "category": "function",
    "text": "getpoints(qr::QuadratureRule)\n\nReturn the points of the quadrature rule.\n\nExamples\n\njulia> qr = QuadratureRule{2, RefTetrahedron}(:legendre, 2);\n\njulia> getpoints(qr)\n3-element Array{Tensors.Tensor{1,2,Float64,2},1}:\n [0.166667, 0.166667]\n [0.166667, 0.666667]\n [0.666667, 0.166667]\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getweights",
    "page": "Utilities",
    "title": "JuAFEM.getweights",
    "category": "function",
    "text": "getweights(qr::QuadratureRule)\n\nReturn the weights of the quadrature rule.\n\nExamples\n\njulia> qr = QuadratureRule{2, RefTetrahedron}(:legendre, 2);\n\njulia> getweights(qr)\n3-element Array{Float64,1}:\n 0.166667\n 0.166667\n 0.166667\n\n\n\n"
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
    "category": "function",
    "text": "Return the number of base functions for an Interpolation or Values object.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getdim",
    "page": "Utilities",
    "title": "JuAFEM.getdim",
    "category": "function",
    "text": "Return the dimension of an Interpolation\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getrefshape",
    "page": "Utilities",
    "title": "JuAFEM.getrefshape",
    "category": "function",
    "text": "Return the reference shape of an Interpolation\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getorder",
    "page": "Utilities",
    "title": "JuAFEM.getorder",
    "category": "function",
    "text": "Return the polynomial order of the Interpolation\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#Interpolation-1",
    "page": "Utilities",
    "title": "Interpolation",
    "category": "section",
    "text": "getnbasefunctions\ngetdim\ngetrefshape\ngetorder"
},

{
    "location": "lib/utility_functions.html#JuAFEM.reinit!",
    "page": "Utilities",
    "title": "JuAFEM.reinit!",
    "category": "function",
    "text": "reinit!(cv::CellValues, x::Vector)\nreinit!(bv::FaceValues, x::Vector, face::Int)\n\nUpdate the CellValues/FaceValues object for a cell or face with coordinates x. The derivatives of the shape functions, and the new integration weights are computed.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getnquadpoints",
    "page": "Utilities",
    "title": "JuAFEM.getnquadpoints",
    "category": "function",
    "text": "getnquadpoints(fe_v::Values)\n\nReturn the number of quadrature points for the Values object.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getdetJdV",
    "page": "Utilities",
    "title": "JuAFEM.getdetJdV",
    "category": "function",
    "text": "getdetJdV(fe_v::Values, q_point::Int)\n\nReturn the product between the determinant of the Jacobian and the quadrature point weight for the given quadrature point: det(J(mathbfx)) w_q\n\nThis value is typically used when one integrates a function on a finite element cell or face as\n\nintlimits_Omega f(mathbfx) d Omega approx sumlimits_q = 1^n_q f(mathbfx_q) det(J(mathbfx)) w_q intlimits_Gamma f(mathbfx) d Gamma approx sumlimits_q = 1^n_q f(mathbfx_q) det(J(mathbfx)) w_q\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.shape_value",
    "page": "Utilities",
    "title": "JuAFEM.shape_value",
    "category": "function",
    "text": "shape_value(fe_v::Values, q_point::Int, base_function::Int)\n\nReturn the value of shape function base_function evaluated in quadrature point q_point.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.shape_gradient",
    "page": "Utilities",
    "title": "JuAFEM.shape_gradient",
    "category": "function",
    "text": "shape_gradient(fe_v::Values, q_point::Int, base_function::Int)\n\nReturn the gradient of shape function base_function evaluated in quadrature point q_point.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.shape_symmetric_gradient",
    "page": "Utilities",
    "title": "JuAFEM.shape_symmetric_gradient",
    "category": "function",
    "text": "shape_symmetric_gradient(fe_v::Values, q_point::Int, base_function::Int)\n\nReturn the symmetric gradient of shape function base_function evaluated in quadrature point q_point.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.shape_divergence",
    "page": "Utilities",
    "title": "JuAFEM.shape_divergence",
    "category": "function",
    "text": "shape_divergence(fe_v::Values, q_point::Int, base_function::Int)\n\nReturn the divergence of shape function base_function evaluated in quadrature point q_point.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.function_value",
    "page": "Utilities",
    "title": "JuAFEM.function_value",
    "category": "function",
    "text": "function_value(fe_v::Values, q_point::Int, u::AbstractVector)\n\nCompute the value of the function in a quadrature point. u is a vector with values for the degrees of freedom. For a scalar valued function, u contains scalars. For a vector valued function, u can be a vector of scalars (for use of VectorValues) or u can be a vector of Vecs (for use with ScalarValues).\n\nThe value of a scalar valued function is computed as u(mathbfx) = sumlimits_i = 1^n N_i (mathbfx) u_i where u_i are the value of u in the nodes. For a vector valued function the value is calculated as mathbfu(mathbfx) = sumlimits_i = 1^n N_i (mathbfx) mathbfu_i where mathbfu_i are the nodal values of mathbfu.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.function_gradient",
    "page": "Utilities",
    "title": "JuAFEM.function_gradient",
    "category": "function",
    "text": "function_scalar_gradient(fe_v::Values{dim}, q_point::Int, u::AbstractVector)\n\nCompute the gradient of the function in a quadrature point. u is a vector with values for the degrees of freedom. For a scalar valued function, u contains scalars. For a vector valued function, u can be a vector of scalars (for use of VectorValues) or u can be a vector of Vecs (for use with ScalarValues).\n\nThe gradient of a scalar function is computed as mathbfnabla u(mathbfx) = sumlimits_i = 1^n mathbfnabla N_i (mathbfx) u_i where u_i are the nodal values of the function. For a vector valued function the gradient is computed as mathbfnabla mathbfu(mathbfx) = sumlimits_i = 1^n mathbfnabla N_i (mathbfx) otimes mathbfu_i where mathbfu_i are the nodal values of mathbfu.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.function_symmetric_gradient",
    "page": "Utilities",
    "title": "JuAFEM.function_symmetric_gradient",
    "category": "function",
    "text": "function_symmetric_gradient(fe_v::Values, q_point::Int, u::AbstractVector)\n\nCompute the symmetric gradient of the function, see function_gradient. Return a SymmetricTensor.\n\nThe symmetric gradient of a scalar function is computed as left mathbfnabla  mathbfu(mathbfx_q) right^textsym =  sumlimits_i = 1^n  frac12 left mathbfnabla N_i (mathbfx_q) otimes mathbfu_i + mathbfu_i  otimes  mathbfnabla N_i (mathbfx_q) right where mathbfu_i are the nodal values of the function.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.function_divergence",
    "page": "Utilities",
    "title": "JuAFEM.function_divergence",
    "category": "function",
    "text": "function_divergence(fe_v::Values, q_point::Int, u::AbstractVector)\n\nCompute the divergence of the vector valued function in a quadrature point.\n\nThe divergence of a vector valued functions in the quadrature point mathbfx_q) is computed as mathbfnabla cdot mathbfu(mathbfx_q) = sumlimits_i = 1^n mathbfnabla N_i (mathbfx_q) cdot mathbfu_i where mathbfu_i are the nodal values of the function.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.spatial_coordinate",
    "page": "Utilities",
    "title": "JuAFEM.spatial_coordinate",
    "category": "function",
    "text": "spatial_coordinate(fe_v::Values{dim}, q_point::Int, x::AbstractVector)\n\nCompute the spatial coordinate in a quadrature point. x contains the nodal coordinates of the cell.\n\nThe coordinate is computed, using the geometric interpolation, as mathbfx = sumlimits_i = 1^n M_i (mathbfx) mathbfhatx_i\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#CellValues-1",
    "page": "Utilities",
    "title": "CellValues",
    "category": "section",
    "text": "reinit!\ngetnquadpoints\ngetdetJdV\n\nshape_value\nshape_gradient\nshape_symmetric_gradient\nshape_divergence\n\nfunction_value\nfunction_gradient\nfunction_symmetric_gradient\nfunction_divergence\nspatial_coordinate"
},

{
    "location": "lib/utility_functions.html#JuAFEM.getcurrentface",
    "page": "Utilities",
    "title": "JuAFEM.getcurrentface",
    "category": "function",
    "text": "getcurrentface(fv::FaceValues)\n\nReturn the current active face of the FaceValues object (from last reinit!).\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#FaceValues-1",
    "page": "Utilities",
    "title": "FaceValues",
    "category": "section",
    "text": "All of the methods for CellValues apply for FaceValues as well. In addition, there are some methods that are unique for FaecValues:getcurrentface"
},

{
    "location": "lib/utility_functions.html#JuAFEM.start_assemble",
    "page": "Utilities",
    "title": "JuAFEM.start_assemble",
    "category": "function",
    "text": "start_assemble([N=0]) -> Assembler\n\nCall before starting an assembly.\n\nReturns an Assembler type that is used to hold the intermediate data before an assembly is finished.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.assemble!",
    "page": "Utilities",
    "title": "JuAFEM.assemble!",
    "category": "function",
    "text": "assemble!(a, Ke, edof)\n\nAssembles the element matrix Ke into a.\n\n\n\nassemble!(g, ge, edof)\n\nAssembles the element residual ge into the global residual vector g.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#JuAFEM.end_assemble",
    "page": "Utilities",
    "title": "JuAFEM.end_assemble",
    "category": "function",
    "text": "end_assemble(a::Assembler) -> K\n\nFinalizes an assembly. Returns a sparse matrix with the assembled values.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#Assembling-1",
    "page": "Utilities",
    "title": "Assembling",
    "category": "section",
    "text": "start_assemble\nassemble!\nend_assemble"
},

{
    "location": "lib/utility_functions.html#WriteVTK.vtk_grid",
    "page": "Utilities",
    "title": "WriteVTK.vtk_grid",
    "category": "function",
    "text": "vtk_grid(filename::AbstractString, grid::Grid)\n\nCreate a unstructured VTK grid from a Grid. Return a DatasetFile which data can be appended to, see vtk_point_data and vtk_cell_data.\n\n\n\n"
},

{
    "location": "lib/utility_functions.html#VTK-1",
    "page": "Utilities",
    "title": "VTK",
    "category": "section",
    "text": "vtk_grid"
},

{
    "location": "examples/heat_equation.html#",
    "page": "Heat equation",
    "title": "Heat equation",
    "category": "page",
    "text": ""
},

{
    "location": "examples/heat_equation.html#Heat-equation-1",
    "page": "Heat equation",
    "title": "Heat equation",
    "category": "section",
    "text": "cp(joinpath(\"/home/travis/.julia/v0.6/JuAFEM/test/../docs/src/examples\", \"..\", \"..\", \"..\", \"examples\", \"figures\", \"heat_square.png\"), \"heat_equation.png\")(Image: )"
},

{
    "location": "examples/heat_equation.html#Introduction-1",
    "page": "Heat equation",
    "title": "Introduction",
    "category": "section",
    "text": "The heat equation is the \"Hello, world!\" equation of finite elements. Here we solve the equation on a unit square, with a uniform internal source. The strong form of the (linear) heat equation is given by- nabla cdot (k nabla u) = f  quad x in Omegawhere u is the unknown temperature field, k the heat conductivity, f the heat source and Omega the domain. For simplicity we set f = 1 and k = 1. We will consider homogeneous Dirichlet boundary conditions such thatu(x) = 0 quad x in partial Omegawhere partial Omega denotes the boundary of Omega.The resulting weak form is given byint_Omega nabla v cdot nabla u  dOmega = int_Omega v  dOmegawhere v is a suitable test function."
},

{
    "location": "examples/heat_equation.html#Commented-program-1",
    "page": "Heat equation",
    "title": "Commented program",
    "category": "section",
    "text": "Now we solve the problem in JuAFEM. What follows is a program spliced with comments. The full program, without comments, can be found in the next section.First we load the JuAFEM package.using JuAFEMWe start  generating a simple grid with 20x20 quadrilateral elements using generate_grid. The generator defaults to the unit square, so we don\'t need to specify the corners of the domain.grid = generate_grid(Quadrilateral, (20, 20))"
},

{
    "location": "examples/heat_equation.html#Trial-and-test-functions-1",
    "page": "Heat equation",
    "title": "Trial and test functions",
    "category": "section",
    "text": "A CellValues facilitates the process of evaluating values and gradients of test and trial functions (among other things). Since the problem is a scalar problem we will use a CellScalarValues object. To define this we need to specify an interpolation space for the shape functions. We use Lagrange functions (both for interpolating the function and the geometry) based on the reference \"cube\". We also define a quadrature rule based on the same reference cube. We combine the interpolation and the quadrature rule to a CellScalarValues object.dim = 2\nip = Lagrange{dim, RefCube, 1}()\nqr = QuadratureRule{dim, RefCube}(2)\ncellvalues = CellScalarValues(qr, ip)"
},

{
    "location": "examples/heat_equation.html#Degrees-of-freedom-1",
    "page": "Heat equation",
    "title": "Degrees of freedom",
    "category": "section",
    "text": "Next we need to define a DofHandler, which will take care of numbering and distribution of degrees of freedom for our approximated fields. We create the DofHandler and then add a single field called u. Lastly we close! the DofHandler, it is now that the dofs are distributed for all the elements.dh = DofHandler(grid)\npush!(dh, :u, 1)\nclose!(dh)Now that we have distributed all our dofs we can create our tangent matrix, using create_sparsity_pattern. This function returns a sparse matrix with the correct elements stored.K = create_sparsity_pattern(dh)We can inspect the pattern using the spy function from UnicodePlots.jl. By default the stored values are set to 0, so we first need to fill the stored values, e.g. K.nzval with something meaningful.fill!(K.nzval, 1.0)\nspy(K)which results in             Sparsity Pattern\n      ┌──────────────────────────────┐\n    1 │⠻⣦⡈⢧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ > 0\n      │⠦⣌⡻⣮⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ < 0\n      │⠀⠀⠉⠻⣿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│\n      │⠀⠀⠀⠀⠈⠻⣿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│\n      │⠀⠀⠀⠀⠀⠀⠈⠻⣿⣿⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│\n      │⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⣿⣿⣦⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│\n      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠿⣿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│\n      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│\n      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│\n      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣿⣿⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀│\n      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⣿⣿⣦⡄⠀⠀⠀⠀⠀⠀│\n      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠿⣿⣿⣦⡀⠀⠀⠀⠀│\n      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣿⣿⣦⡀⠀⠀│\n      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣿⣿⣦⡀│\n  441 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣿⣿│\n      └──────────────────────────────┘\n      1                            441\n                 nz = 3721"
},

{
    "location": "examples/heat_equation.html#Boundary-conditions-1",
    "page": "Heat equation",
    "title": "Boundary conditions",
    "category": "section",
    "text": "In JuAFEM constraints like Dirichlet boundary conditions are handled by a ConstraintHandler.ch = ConstraintHandler(dh)Next we need to add constraints to ch. For this problem we define homogeneous Dirichlet boundary conditions on the whole boundary, i.e. the union of all the face sets on the boundary.∂Ω = union(getfaceset.(grid, [\"left\", \"right\", \"top\", \"bottom\"])...)Now we are set up to define our constraint. We specify which field the condition is for, and our combined face set ∂Ω. The last argument is a function which takes the spatial coordinate x and the current time t and returns the prescribed value. In this case it is trivial – no matter what x and t we return 0. When we have specified our contraint we add! it to ch.dbc = DirichletBoundaryCondition(:u, ∂Ω, (x, t) -> 0)\nadd!(ch, dbc)We also need to close! and update! our boundary conditions. When we call close! the dofs which will be constrained by the boundary conditions are calculated and stored in our ch object. Since the boundary conditions are, in this case, independent of time we can update! them directly with e.g. t = 0.close!(ch)\nupdate!(ch, 0.0)"
},

{
    "location": "examples/heat_equation.html#Assembling-the-linear-system-1",
    "page": "Heat equation",
    "title": "Assembling the linear system",
    "category": "section",
    "text": "Now we have all the pieces needed to assemble the linear system, K u = f. We define a function, doassemble to do the assembly, which takes our cellvalues, the sparse matrix and our DofHandler as input arguments. The function returns the assembled stiffness matrix, and the force vector.function doassemble(cellvalues::CellScalarValues{dim}, K::SparseMatrixCSC, dh::DofHandler) where {dim}We allocate the element stiffness matrix and element force vector just once before looping over all the cells instead of allocating them every time in the loop.    n_basefuncs = getnbasefunctions(cellvalues)\n    Ke = zeros(n_basefuncs, n_basefuncs)\n    fe = zeros(n_basefuncs)Next we define the global force vector f and use that and the stiffness matrix K and create an assembler. The assembler is just a thin wrapper around f and K and some extra storage to make the assembling faster.    f = zeros(ndofs(dh))\n    assembler = start_assemble(K, f)It is now time to loop over all the cells in our grid. We do this by iterating over a CellIterator. The iterator caches some useful things for us, for example the nodal coordinates for the cell, and the local degrees of freedom.    @inbounds for cell in CellIterator(dh)Always remember to reset the element stiffness matrix and force vector since we reuse them for all elements.        fill!(Ke, 0)\n        fill!(fe, 0)For each cell we also need to reinitialize the cached values in cellvalues.        reinit!(cellvalues, cell)It is now time to loop over all the quadrature points in the cell and assemble the contribution to Ke and fe. The integration weight can be queried from cellvalues by getdetJdV.        for q_point in 1:getnquadpoints(cellvalues)\n            dΩ = getdetJdV(cellvalues, q_point)For each quadrature point we loop over all the (local) shape functions. We need the value and gradient of the testfunction v and also the gradient of the trial function u. We get all of these from cellvalues.            for i in 1:n_basefuncs\n                v  = shape_value(cellvalues, q_point, i)\n                ∇v = shape_gradient(cellvalues, q_point, i)\n                fe[i] += v * dΩ\n                for j in 1:n_basefuncs\n                    ∇u = shape_gradient(cellvalues, q_point, j)\n                    Ke[i, j] += (∇v ⋅ ∇u) * dΩ\n                end\n            end\n        endThe last step in the element loop is to assemble Ke and fe into the global K and f with assemble!.        assemble!(assembler, celldofs(cell), fe, Ke)\n    end\n    return K, f\nend"
},

{
    "location": "examples/heat_equation.html#Solution-of-the-system-1",
    "page": "Heat equation",
    "title": "Solution of the system",
    "category": "section",
    "text": "The last step is to solve the system. First we call doassemble to obtain the global stiffness matrix K and force vector f.K, f = doassemble(cellvalues, K, dh)To account for the boundary conditions we use the apply! function. This modifies elements in K and f respectively, such that we can get the correct solution vector u by using \\.apply!(K, f, ch)\nu = K \\ f"
},

{
    "location": "examples/heat_equation.html#Exporting-to-VTK-1",
    "page": "Heat equation",
    "title": "Exporting to VTK",
    "category": "section",
    "text": "To visualize the result we export the grid and our field u to a VTK-file, which can be viewed in e.g. ParaView.vtk_grid(\"heat_equation\", dh) do vtk\n    vtk_point_data(vtk, dh, u)\nend"
},

{
    "location": "examples/heat_equation.html#heat_equation-plain-program-1",
    "page": "Heat equation",
    "title": "Plain program",
    "category": "section",
    "text": "Below follows a version of the program without any comments. The file is also available here: [heat_equation.jl]include(\"/home/travis/.julia/v0.6/JuAFEM/test/../docs/generate.jl\")\nwrite(\"heat_equation.jl\", extract_code(\"/home/travis/.julia/v0.6/JuAFEM/test/../docs/src/examples/heat_equation.jl\"))using JuAFEM\n\ngrid = generate_grid(Quadrilateral, (20, 20))\n\ndim = 2\nip = Lagrange{dim, RefCube, 1}()\nqr = QuadratureRule{dim, RefCube}(2)\ncellvalues = CellScalarValues(qr, ip)\n\ndh = DofHandler(grid)\npush!(dh, :u, 1)\nclose!(dh)\n\nK = create_sparsity_pattern(dh)\n\nch = ConstraintHandler(dh)\n∂Ω = union(getfaceset.(grid, [\"left\", \"right\", \"top\", \"bottom\"])...)\ndbc = DirichletBoundaryCondition(:u, ∂Ω, (x, t) -> 0)\nadd!(ch, dbc)\nclose!(ch)\nupdate!(ch, 0.0)\n\nfunction doassemble(cellvalues::CellScalarValues{dim}, K::SparseMatrixCSC, dh::DofHandler) where {dim}\n    n_basefuncs = getnbasefunctions(cellvalues)\n    Ke = zeros(n_basefuncs, n_basefuncs)\n    fe = zeros(n_basefuncs)\n\n    f = zeros(ndofs(dh))\n    assembler = start_assemble(K, f)\n\n    @inbounds for cell in CellIterator(dh)\n        fill!(Ke, 0)\n        fill!(fe, 0)\n\n        reinit!(cellvalues, cell)\n\n        for q_point in 1:getnquadpoints(cellvalues)\n            dΩ = getdetJdV(cellvalues, q_point)\n            for i in 1:n_basefuncs\n                v  = shape_value(cellvalues, q_point, i)\n                ∇v = shape_gradient(cellvalues, q_point, i)\n                fe[i] += v * dΩ\n                for j in 1:n_basefuncs\n                    ∇u = shape_gradient(cellvalues, q_point, j)\n                    Ke[i, j] += (∇v ⋅ ∇u) * dΩ\n                end\n            end\n        end\n\n        assemble!(assembler, celldofs(cell), fe, Ke)\n    end\n    return K, f\nend\n\nK, f = doassemble(cellvalues, K, dh)\napply!(K, f, ch)\nu = K \\ f\n\nvtk_grid(\"heat_equation\", dh) do vtk\n    vtk_point_data(vtk, dh, u)\nendThis page was rendered automatically based on heat_equation.jl"
},

]}
