# LinkedIn Queens Solver

A minimal brute‑force solver for LinkedIn’s Queens puzzle. No frills—just a clean backtracking implementation guided by color‑region size.

---

## 🗒️ Overview

This solver works in clear, logical stages to tackle the Queens puzzle from a screenshot:

1. **Image loading & conversion**

   * Read `game.png` using OpenCV in BGR format.
   * Convert to RGB so color values map correctly to human vision and clustering.

2. **Grid slicing**

   * Determine cell height and width by dividing image dimensions by `grid_size`.
   * Slice the image into an N×N grid of equal rectangles.

3. **Average color extraction**

   * For each cell, flatten its pixels into a list of `[R, G, B]`.
   * Compute the mean across all pixels, yielding one representative color per cell.

4. **K‑Means clustering**

   * Cluster the N average colors into `grid_size` groups.
   * Each cluster corresponds to one unique puzzle region (color label).
   * Sort cluster centers by overall brightness and remap labels to `C0…C{N-1}` so the easiest regions emerge first.

5. **Board model construction**

   * Build a 2D `board` list: each entry is a dict `{ Color: 'Cx', state: 'eligible' }`.
   * Every cell starts as a potential queen placement.

6. **Eradication routine** (`post_queen_eradication`)
   When placing a queen at `(r, c)`, the function:

   * Marks `(r,c)` as `'queen'`.
   * Marks all other cells in row `r` and column `c` as `'ineligible'`.
   * Marks every other cell of the same color region as `'ineligible'`.
   * Marks the 8 neighboring cells around `(r,c)` as `'ineligible'`.
   * Captures a board snapshot (frame) after each change for animation.

7. **Automatic obvious placements** (`do_obv_placing`)

   * Repeatedly scans rows, columns, and color regions.
   * If any has exactly one `'eligible'` cell, places a queen there immediately.
   * Continues until no further automatic placements are possible.

8. **Invalid-state detection** (`inv_board_chk`)

   * Checks each row, column, and color region.
   * If any has zero cells left in `'eligible'` or `'queen'` state, the branch is invalid.

9. **Backtracking search** (`solve_queens`)

   * Start with the initial board and zero queens.
   * Apply obvious placements.
   * If the board is invalid, backtrack.
   * If N queens are placed, solution found.
   * Otherwise, pick the color region with the fewest eligible cells (smallest branching factor).
   * Try placing a queen in each eligible cell of that region, recursing until success or exhaustion.

10. **Animation & output**

    * Record all board states into `frames`.
    * Print total solve time.
    * Save the final board image.
    * Write `queens_solver.mp4` showing the entire placement process.

## ⚙️ Requirements

* Python 3.7+
* OpenCV (`cv2`)
* NumPy
* scikit-learn
* imageio
* Matplotlib

You can install all with:

```bash
pip install opencv-python numpy scikit-learn imageio matplotlib
```

## 🚀 Usage

1. Put your board screenshot in the same folder as `game.png`.
2. Run:

   ```bash
   python queens_solver.py
   ```
3. Watch the console for timing and see `queens_solver.mp4` appear.

## 🔍 How It Works

* **Color extraction**: Average pixels per cell → cluster → label.
* **Board model**: Each cell tracks its color and state.
* **Eradication**: Placing a queen marks row, column, region, neighbors ineligible.
* **Heuristic**: Always branch on the smallest region.
* **Backtracking**: Deep‑copy board + frames, undo on invalid.

## 📝 Files

* `queens_solver.py` — all logic in one script.
* `game.png`         — input screenshot.
* `queens_solver.mp4` — output animation.

## 📄 License

MIT License. Copy, modify, share—no strict rules.
