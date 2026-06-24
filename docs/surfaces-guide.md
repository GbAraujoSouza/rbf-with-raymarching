# Adding New Surfaces to the RBF Raymarcher

This guide explains the mathematical intuition and logic needed to implement new surfaces for the Radial Basis Function (RBF) raymarching engine.

The core principle for adding any new shape is always the same: **generate a set of surface points ($P$) and their corresponding outward-pointing normal vectors ($\hat{n}$)**. 

Once you have $P$ and $\hat{n}$, you use the Hermite interpolation (normal offset) method to constrain the RBF:
1.  **Surface point**: Add $P$ with a target of `0`.
2.  **Outer offset point**: Add $P + \epsilon \cdot \hat{n}$ with a target of $+\epsilon$.
3.  **Inner offset point**: Add $P - \epsilon \cdot \hat{n}$ with a target of $-\epsilon$.
*(Where $\epsilon$ is your configured `normalOffset`)*

Below is the logic for several common and interesting shapes.

---

## 1. The Torus (Doughnut)

A torus is built from two circles: a large circular path, and a smaller circle (the "tube") that sweeps around that path. Any point on its surface can be found using two angles:
*   **$u$**: The angle sweeping around the main vertical axis (from $0$ to $2\pi$).
*   **$v$**: The angle sweeping around the tube itself (from $0$ to $2\pi$).

### Position
Using `majorRadius` ($R$) for the distance to the tube center, and `minorRadius` ($r$) for the tube's thickness:
*   $X = (R + r \cdot \cos(v)) \cdot \cos(u)$
*   $Z = (R + r \cdot \cos(v)) \cdot \sin(u)$
*   $Y = r \cdot \sin(v)$

### Normal
The normal is simply the 2D outward direction of the tube cross-section, rotated around the Y-axis by angle $u$:
*   $N_x = \cos(v) \cdot \cos(u)$
*   $N_z = \cos(v) \cdot \sin(u)$
*   $N_y = \sin(v)$

---

## 2. The Cube / Box

A cube is made of 6 flat square faces. Instead of one continuous parametric equation, you handle it as 6 separate loops.

### Sampling & Normals
Define the half-size of your cube as $L$. For each face, create a 2D grid of points ranging from $-L$ to $L$ on the appropriate axes. The normal for a flat face is constant.

*   **Right face** ($X = L$): 
    *   Position: $(L, y, z)$ 
    *   Normal: `(1, 0, 0)`
*   **Left face** ($X = -L$): Normal `(-1, 0, 0)`
*   **Top face** ($Y = L$): Normal `(0, 1, 0)`
*   **Bottom face** ($Y = -L$): Normal `(0, -1, 0)`
*   **Front face** ($Z = L$): Normal `(0, 0, 1)`
*   **Back face** ($Z = -L$): Normal `(0, 0, -1)`

---

## 3. The Cylinder

A cylinder is made of three distinct parts: the curved "tube" around the side, and two flat circular "caps" on the top and bottom.

### The Tube (Side)
*   **Sampling**: Loop for the angle $u \in [0, 2\pi]$ and the height $y \in [-H/2, H/2]$.
*   **Position**: $(R \cdot \cos(u), y, R \cdot \sin(u))$
*   **Normal**: The normal only points outwards horizontally: `(cos(u), 0, sin(u))`.

### The Top Cap
*   **Sampling**: Create a 2D grid of points on a flat circle of radius $R$ at height $+H/2$.
*   **Position**: $(x, +H/2, z)$
*   **Normal**: Completely flat pointing up: `(0, 1, 0)`.

### The Bottom Cap
*   **Position**: $(x, -H/2, z)$
*   **Normal**: Completely flat pointing down: `(0, -1, 0)`.

---

## 4. Two Disjoint Spheres

This tests how the RBF field merges or separates distinct objects (the "metaball" effect).

### Logic
Run your standard sphere generation logic *twice*, but shift the center point.
*   **Sphere 1**: Shift all points to the left by distance $D$. $P_{new} = (P_x - D, P_y, P_z)$. 
*   **Sphere 2**: Shift all points to the right by distance $D$. $P_{new} = (P_x + D, P_y, P_z)$. 
*   **Normal**: The normal vector $\hat{n}$ remains exactly the same as the original centered sphere! Shifting an object doesn't change the direction its surface is pointing.

---

## 5. Mathematical Implicit Shapes (Heart / Star)

Complex mathematical shapes are usually defined by an **Implicit Equation** (e.g., $F(x,y,z) = 0$). To generate constraint samples from an equation, use **Rejection Sampling**:

1.  **Sampling**: Generate random 3D points inside a bounding box. Plug each point into $F(x,y,z)$. If the result is very close to $0$ (e.g., $|F| < 0.01$), keep the point because it lies on the surface.
2.  **Normal**: The normal of an implicit equation at any point is its **Gradient** ($\nabla F$). Calculate the partial derivatives with respect to X, Y, and Z. The normal vector at point $(x,y,z)$ is the normalized gradient: `normalize(dF/dx, dF/dy, dF/dz)`.

*(Tip: Writing partial derivatives can be tedious. It is often easier to find an `.obj` file of the shape and parse the vertices and vertex normals directly.)*

---

## 6. Real Objects (OBJ Meshes)

To fit an RBF to a point cloud of a real object (like the Stanford Bunny or Dragon), you must parse both the vertices (`v`) and their vertex normals (`vn`).

1.  **Parse**: Extract both position $P$ and normal $\hat{n}$ for every vertex.
2.  **Sample**: Apply the Hermite interpolation method exactly as described above. The RBF will perfectly capture the shape because it knows exactly which direction is "out" and which direction is "in" at every single vertex.
