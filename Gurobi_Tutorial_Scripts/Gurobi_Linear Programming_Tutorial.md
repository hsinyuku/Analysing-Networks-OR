# mathematical programming

**Three steps:**

1. formulation of real-world problems in detailed mathematical terms (models)
2. development of techniques for solving these models (algorithms)
3. use software and hardware to develop applications

- different from computer programming
- common feature: optimisation

In **Gurobi**, the user only needs to have a mathematical formulation (linear programming model or mixed integer programming model) of the problem; the solving will be done by Gurobi in the background

# formulation of the problem

The furniture problem:

- **decision variables** are the numbers of products $x_1, x_2$ of each type to produce$x_1, x_2 \ge 0$ (assuming fractional quantities of products can be sold)

- trying to **maximise** revenue from two kinds of products: $x_1p_1+x_2p_2$ (**objective function**)

- **constraints** are the resources that the company has to produce the products

  - different resources $K_1, K_2$ are available

  - different products use different amounts of the resources: $k_1^1, k_1^2, k_2^1, k_2^2$

  - the constraints are thus $r_1^1*x_1 + k_1^2*x_2 < K_1$ (and similar for $R_2$)

    

$$
\text{max}\quad x_1p_1+x_2p_2 \\
  \text{subject to constraints}\\ k_1^1*x_1 + k_1^2*x_2 < K_1, \\ k_2^1*x_1 + k_2^2*x_2 < K_2, \\ x_1, x_2 \ge 0
$$


  -  using sets for notation:

  - the set "products" maps an index with a corresponding product
  - the set "resources" does the same for resources
  - parameters are defined over sets
  - **technology coefficients** denote the number of resources used to produce a product

      - the first index of a technology coefficient refers to resources, the second index to products
  - a **matrix of technology coefficients** can be set up
  - the **objective function** is what we try to optimise

  - **parametrisation** allows the separation of data from the model

  - for many products and many resources (more general formulation) - **LP formulation**:

    
    $$
    Max\sum_{j=1}^{n}b_jx_j\\
    \sum_{j=1}^{n}a_{i, j}x_j \le K_i \quad | \quad i = 1, \dots, m \\
    x_j \ge 0 \quad |  \quad j = 1, \dots, n
    $$


## types of programming problems

- **integer programming problems** require $x_j$ to be integers
- **binary programming problems** require $x_j\in\{0, 1\}$

- **mixed integer linear programming problems** have different requirements for different $x_j$
    - here is is possible to have equivalent formulations of a problem
    - the performance of the solver depends on the formulation

## formulation of the problem

- express one of the decision variables as a function of the other
- introduce a slack variable $h_i \quad | \quad i = 1, \dots, m$ to denote how many resources are unused after a certain production plan
- the total resources used can now be expressed as an equality, rather than an inequality:

$$
\sum_{j=1}^{n}a_{i, j}x_j +h_i = K_i \quad | \quad i = 1, \dots, m \text{ denoting different resources}
$$



- the feasible region is called the **polyhedron**
- **graphically**, the optimal solution is the optimum of the objective function where $x_1$ and $x_2$ are in the polyhedron

# fundamental theorem of linear programming



- a <u>**solution**</u> of a LP problem is a set of values of the decision variables that satisfies all the constraints of the problem defined by the polyhedron
- a **<u>corner point solution</u>** is a vertex of the polyhedron defining the feasible region of the LP problem
- an **<u>optimal solution</u>** is a solution of the LP problem that cannot be improved

- <u>**linear programming theorem**</u>:

> If a LP problem has an optimal solution, then there is at least one optimal solution that is a corner point solution

# the simplex method

- the simplex method is an algorithm to solve LP problems
- general problem: how to traverse all the vertices of the polyhedron?

- convert the LP problem into a **standard form**:
  $$
  \text{maximise: }\sum_{j=1}^{n}b_jx_j\\
  \sum_{j=1}^{n}a_{i, j}x_j + h_i = K_i \\
  x_j, h_i \ge 0  \\
  j = 1, \dots, n \text{ (products)} \quad | \quad i = 1, \dots, m  \text{ (resources)}
  $$
  
- values >0 are called basic variables, values =0 are non basic variables

- the **basic feasible solution** is the solution where the non-basic variables are zero

- if $h_i$ are negative, they denote the amount of additional resources to fulfill a certain solution

- a basic infeasible solution is a solution where all but on of $h_i$ are zero, and the last $h_i$ is negative

## the canonical form

- express the basic variables as function of the non-basic variables
- **reduced costs** are the objective function coefficients of the non-basic variables

- **min ratio test** chooses the minimum value of both resource constraints (for a solution where all but one products are zero, this guarantees a solution in which a production does not exceed resource capacities)
  $$
  \min \{\frac{K_1}{a_{1, j}}, \dots, \frac{K_m}{a_{m, j}}\}
  $$


# using Python and Gurobi to solve linear problems

see IPython notebook

# economic interpretation: sensitivity analysis

- associated with an LP optimal solution are **shadow prices** (also called **dual variable** or **marginal values**) for the constraints
- the shadow price of a constraint variable represents the change in the value of the objective function per unit of increase in that variable
- there are shadow prices associated with non-negativity constraints; these shadow prices are called **reduced costs**

## example

- suppose the labor capacity is increased from 450 to 451 hours
- what is the increase in the objective function?
- since the constraints on mahogany and labor define the optimal solution, we can solve the system of equations of the constraints to get the new value of the objective function
- the difference of the new to the old value of the objective function is the shadow price
- We can get 4$ of increased revenue per hour of increased labour capacity.
- The shadow price remains constant over a range of changes of the mahogany capacity, not for any capacity value.

## the simplex method revisited

- the shadow prices are equal to the values of the slack variables in the optimal solution associated with the resources
- the simplex method thus automaticaly gives us the shadow prices

## economic interpretation of shadow prices

- 