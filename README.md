# Linkedin-Queens-Solver


This project reads an image (tight screenshot or screen snippet of the game board) of an *n×n* colored board (no queens or crosses), identifies the distinct colored regions, and then solves a variant of the n‑Queens puzzle on that board. Along the way it records each step and finally writes out an MP4 animation of the entire solving process.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [How It Works (Thought Process)](#how-it-works-thought-process)

   * [1. Image Processing & Clustering](#1-image-processing--clustering)
   * [2. Board Representation](#2-board-representation)
   * [3. Drawing & Animation](#3-drawing--animation)
   * [4. Solving Heuristics](#4-solving-heuristics)
5. [Function Reference](#function-reference)
6. [Example](#example)

---

## Overview

You give the script a clean board image like `game.png`. The code splits it into a grid of squares, clusters the background colors into *n* labels (C0…C<sub>n−1</sub>), and builds an internal board of cells, each with:

* **Color**: one of the cluster labels
* **state**: `'eligible'`, `'ineligible'`, or `'queen'`

It then applies a sequence of elimination rules (inspired from my own thought process while solving the game) until all *n* queens are placed or no further moves are possible. Each change is rendered into a frame. At the end, an MP4 file shows the solving animation.

---

## Installation

1. Make sure you have these packages installed:

   ```bash
   pip install opencv-python numpy scikit-learn matplotlib imageio
   ```
2. Put your board image (e.g. `game.png`) in the same folder.

---

## Usage

```bash
python queens.py  # or run inside an IDE
```

* It will solve step by step and save `out.mp4` in your folder.

---

## How It Works (Thought Process)

### 1. Image Processing & Clustering

* **Split** the input image into an *n×n* grid.
* **Average** the RGB color of each cell.
* **K‑Means** cluster these into *n* clusters, ensuring each cluster label C0…C<sub>n−1</sub> appears exactly once per row/column in a perfect solution.

### 2. Board Representation

* The `board` is a list of *n* rows, each a list of *n* dictionaries:

  ```python
  {
    'Color': 'C3',      # one of the cluster labels
    'state': 'eligible' # or 'ineligible', 'queen'
  }
  ```

### 3. Drawing & Animation

* `draw_board()` paints each cell with a fixed high‑contrast color palette, draws a border, and overlays

  * **Q** for a placed queen
  * **x** for ineligible cells
* Each time the board state changes, we capture a frame.
* At the end, frames are written to `out.mp4` using `imageio` with H.264 codec.

### 4. Solving Heuristics

The main loop applies these rules in order until no changes occur or all queens are placed:

1. **Obvious Placements**

   * If any **row** or **column** has exactly one eligible square, place a queen there.
   * If any **color region** (all cells sharing the same cluster label) has exactly one eligible square, place a queen there.
2. **Candidate Intersection Elimination**

   * For each color region with ≥2 eligibles, compute for each cell the set of other cells that would attack it (row, column, neighbors, same color). Intersect those sets and mark the common ones ineligible.
3. **Line Candidate Intersection**

   * Same idea, but applied per row and per column: intersect attack sets of all eligible squares in that row (or column) and eliminate the intersection.
4. **Consecutive Group Elimination**

   * For every block of *k* consecutive rows (and columns):

     1. If the union of eligible colors in that block is exactly *k*, then cells of those colors **outside** the block are ineligible.
     2. If the union is > *k*, but exactly *k* colors appear only in that block (and nowhere else), then other colors **inside** the block are ineligible.

The solver loops through these steps, capturing frames on every change, until it either places all queens or detects an **impossible** configuration (a row/column/color with no eligible squares left).

---

## Function Reference

Below is a quick summary of the main functions:

* **`average_cell_colors(image, grid_size)`**

  > Splits image into equal squares and returns the average RGB of each.

* **`make_initial_board(image_path, grid_size)`**

  > Loads image, clusters cell colors, and builds the initial board with all squares eligible.

* **`draw_board(board, cell_size, frames=None)`**

  > Renders current board to an image. If `frames` list is provided, appends frame instead of showing it.

* **`place_queen_and_mark(board, row, col, frames)`**

  > Places a queen at the given cell, then marks all attacked/in-region/neighbor squares ineligible and records each frame.

* **`build_eligibility_maps(board)`**

  > Returns two dicts mapping row → {colors still eligible} and column → {colors still eligible}.

* **`do_obvious_placements(board, queen_count, frames)`**

  > Repeatedly places queens in any row/column/region with exactly one eligible cell.

* **`candidate_intersection_elimination(board, frames)`**

  > Eliminates cells based on intersecting attack sets per color region.

* **`line_candidate_intersection_elimination(board, frames)`**

  > Similar elimination but per entire row or column.

* **`eliminate_consecutive_groups(board, frames, max_k)`**

  > Explores k-length runs of consecutive rows/columns to prune ineligible cells globally or inside blocks.

* **`check_impossible(board)`**

  > Detects if any row, column, or color group has lost all eligible squares without placing a queen.

* **`solve_queens_on_board(board)`**

  > Orchestrates the solving process: captures initial frame, loops through elimination steps, and returns final board plus all captured frames.

---


