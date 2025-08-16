# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Copyright (c) [2025] [Roman Tenger]
import argparse
import math
import re
import sys
import msvcrt
from typing import List, Tuple, Set, Dict
from dataclasses import dataclass
from collections import defaultdict

# Tunable parameters for accuracy
GRID_CELL_SIZE = 0.15  # mm (smaller = more accurate but slower)
BOUNDARY_SAMPLE_POINTS = 75  # points per edge (higher = smoother boundary)
BOUNDARY_SHRINK_STEP = 0.05  # mm (smaller = more precise boundary)
BOUNDARY_MAX_SHRINK = 2.5  # mm (larger = can detect wider areas)
POINT_SEARCH_RADIUS = 0.25  # mm (larger = fills more gaps)
LINE_THICKNESS = 0.45  # mm (larger = more overlap between lines)
SCAN_RESOLUTION = 0.1  # mm (smaller = more precise boundary detection)

################################################################################
# Utilities                                                                     #
################################################################################

@dataclass
class Point:
    x: float
    y: float
    
    def __hash__(self):
        return hash((round(self.x, 4), round(self.y, 4)))
    
    def __eq__(self, other):
        return abs(self.x - other.x) < 1e-4 and abs(self.y - other.y) < 1e-4

@dataclass
class Path:
    points: List[Point]
    is_closed: bool = False

@dataclass
class Boundary:
    points: List[Point]
    is_outer: bool = True

class SupportInterface:
    def __init__(self):
        self.paths: List[Path] = []
        self.bounds = None
        self.points: Set[Point] = set()
        self.boundaries: List[Boundary] = []
        self.grid: Dict[Tuple[int, int], bool] = {}  # Grid-based occupancy map
        self.grid_size = GRID_CELL_SIZE
    
    def add_path(self, path: Path):
        if not path.points:
            return
        
        print(f"  Adding path with {len(path.points)} points")
        self.paths.append(path)
        self.points.update(path.points)
        
        # Add points to grid with increased thickness
        for point in path.points:
            cell_x = int(point.x / self.grid_size)
            cell_y = int(point.y / self.grid_size)
            
            # Add more cells around each point for better coverage
            radius_cells = int(LINE_THICKNESS / self.grid_size)
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    # Only add cells that are within LINE_THICKNESS radius
                    if dx*dx + dy*dy <= radius_cells*radius_cells:
                        self.grid[(cell_x + dx, cell_y + dy)] = True
        
        self._update_bounds()
    
    def _update_bounds(self):
        if not self.points:
            return
        min_x = min(p.x for p in self.points)
        max_x = max(p.x for p in self.points)
        min_y = min(p.y for p in self.points)
        max_y = max(p.y for p in self.points)
        self.bounds = (min_x, max_x, min_y, max_y)
    
    def compute_boundaries(self):
        """Compute boundaries using a shrinkwrap approach"""
        if not self.bounds:
            return
        
        min_x, max_x, min_y, max_y = self.bounds
        boundary_points = []
        
        # Helper function to check if a point is near geometry
        def is_near_geometry(x: float, y: float) -> bool:
            cell_x = int(x / self.grid_size)
            cell_y = int(y / self.grid_size)
            radius_cells = int(POINT_SEARCH_RADIUS / self.grid_size)
            
            # Search in a circular pattern
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    if dx*dx + dy*dy <= radius_cells*radius_cells:
                        if self.grid.get((cell_x + dx, cell_y + dy), False):
                            return True
            return False
        
        # Sample points along each edge and shrink them inward
        def sample_edge(start: Point, end: Point, is_vertical: bool):
            points = []
            for i in range(BOUNDARY_SAMPLE_POINTS):
                t = i / (BOUNDARY_SAMPLE_POINTS - 1)
                if is_vertical:
                    x = start.x
                    y = start.y + t * (end.y - start.y)
                    
                    # Shrink horizontally
                    if x == min_x:  # Left edge
                        for dx in range(0, int(BOUNDARY_MAX_SHRINK / BOUNDARY_SHRINK_STEP)):
                            test_x = x + dx * BOUNDARY_SHRINK_STEP
                            if is_near_geometry(test_x, y):
                                x = test_x
                                break
                    else:  # Right edge
                        for dx in range(0, int(BOUNDARY_MAX_SHRINK / BOUNDARY_SHRINK_STEP)):
                            test_x = x - dx * BOUNDARY_SHRINK_STEP
                            if is_near_geometry(test_x, y):
                                x = test_x
                                break
                else:
                    x = start.x + t * (end.x - start.x)
                    y = start.y
                    
                    # Shrink vertically
                    if y == min_y:  # Bottom edge
                        for dy in range(0, int(BOUNDARY_MAX_SHRINK / BOUNDARY_SHRINK_STEP)):
                            test_y = y + dy * BOUNDARY_SHRINK_STEP
                            if is_near_geometry(x, test_y):
                                y = test_y
                                break
                    else:  # Top edge
                        for dy in range(0, int(BOUNDARY_MAX_SHRINK / BOUNDARY_SHRINK_STEP)):
                            test_y = y - dy * BOUNDARY_SHRINK_STEP
                            if is_near_geometry(x, test_y):
                                y = test_y
                                break
                
                points.append(Point(x, y))
            return points
        
        # Sample all four edges with overlap for better corner detection
        boundary_points.extend(sample_edge(Point(min_x, min_y), Point(min_x, max_y), True))  # Left
        boundary_points.extend(sample_edge(Point(min_x, max_y), Point(max_x, max_y), False))  # Top
        boundary_points.extend(sample_edge(Point(max_x, max_y), Point(max_x, min_y), True))  # Right
        boundary_points.extend(sample_edge(Point(max_x, min_y), Point(min_x, min_y), False))  # Bottom
        
        # Create outer boundary
        self.boundaries = [Boundary(boundary_points)]
        print(f"  Created boundary with {len(boundary_points)} points")

def parse_slicer_settings(lines: List[str]):
    out = {"layer_height": 0.2, "extrusion_width": 0.45, "filament_diameter": 1.75}
    for ln in lines[:300]:
        if ln.startswith(";") and "=" in ln:
            k, v = map(str.strip, ln[1:].split("=", 1))
            m = re.search(r"[-+]?[0-9]*\.?[0-9]+", v)
            if not m:
                continue
            num = float(m.group())
            k = k.lower()
            if "layer_height" in k:
                out["layer_height"] = num
            elif any(t in k for t in ("extrusion_width", "line_width")):
                out["extrusion_width"] = num
            elif "filament_diameter" in k:
                out["filament_diameter"] = num
    return out

def extract_interface_sections(lines: List[str]) -> Dict[int, List[SupportInterface]]:
    sections: Dict[int, List[SupportInterface]] = defaultdict(list)
    current_interface = None
    current_path = None
    last_point = None
    in_interface = False
    interface_count = 0
    
    for i, line in enumerate(lines):
        if ";TYPE:Support material interface" in line:
            print(f"\nFound support interface section at line {i}")
            in_interface = True
            current_interface = SupportInterface()
            current_path = None
            last_point = None
            interface_count += 1
        elif in_interface and (";TYPE:" in line or ";LAYER_CHANGE" in line):
            if current_path is not None and len(current_path.points) > 1:
                current_interface.add_path(current_path)
            
            if current_interface and current_interface.points:
                # Find the next layer change
                layer_idx = i
                while layer_idx < len(lines):
                    if ";LAYER_CHANGE" in lines[layer_idx]:
                        break
                    layer_idx += 1
                if layer_idx < len(lines):
                    sections[layer_idx].append(current_interface)
                    print(f"Added interface section to layer {layer_idx} with {len(current_interface.boundaries)} boundaries")
                    if current_interface.bounds:
                        min_x, max_x, min_y, max_y = current_interface.bounds
                        print(f"  Bounds: X[{min_x:.2f}, {max_x:.2f}] Y[{min_y:.2f}, {max_y:.2f}]")
                        print(f"  Total points: {len(current_interface.points)}")
                        print(f"  Grid cells: {len(current_interface.grid)}")
            
            in_interface = False
            current_interface = None
            current_path = None
            last_point = None
        elif in_interface and line.startswith("G1 "):
            x_match = re.search(r"X([-+]?[0-9]*\.?[0-9]+)", line)
            y_match = re.search(r"Y([-+]?[0-9]*\.?[0-9]+)", line)
            e_match = re.search(r"E([-+]?[0-9]*\.?[0-9]+)", line)
            
            if x_match and y_match and e_match and float(e_match.group(1)) > 0:  # Only consider positive extrusion
                point = Point(float(x_match.group(1)), float(y_match.group(1)))
                if current_path is None:
                    current_path = Path([point])
                    last_point = point
                else:
                    if distance(point, last_point) > 1.0:  # Gap detected, start new path
                        if len(current_path.points) > 1:
                            current_interface.add_path(current_path)
                        current_path = Path([point])
                    else:
                        current_path.points.append(point)
                    last_point = point
    
    print(f"\nSummary:")
    print(f"Found {interface_count} support interface sections")
    print(f"Created {sum(len(sections[k]) for k in sections)} interface objects")
    print(f"Across {len(sections)} layers")
    
    return sections

def distance(p1: Point, p2: Point) -> float:
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

def point_in_polygon(point: Point, polygon: List[Point]) -> bool:
    """Ray casting algorithm to determine if point is inside polygon"""
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        if ((polygon[i].y > point.y) != (polygon[j].y > point.y) and
            point.x < (polygon[j].x - polygon[i].x) * (point.y - polygon[i].y) /
                     (polygon[j].y - polygon[i].y) + polygon[i].x):
            inside = not inside
        j = i
    return inside

def calculate_extrusion(length: float, height: float, width: float, flow: float, filament_diameter: float) -> float:
    # Calculate volume of plastic needed
    volume = length * height * width
    # Convert to length of filament
    filament_area = math.pi * (filament_diameter/2)**2
    return (volume / filament_area) * flow

def shrink_boundary(boundary: List[Point], shrink_distance: float) -> List[Point]:
    """Shrink a boundary inward by the specified distance"""
    if not boundary or len(boundary) < 3:
        return boundary
    
    def get_normal(p1: Point, p2: Point) -> Tuple[float, float]:
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        length = math.sqrt(dx*dx + dy*dy)
        if length < 1e-6:
            return (0, 0)
        # Return normalized vector rotated 90 degrees inward
        return (-dy/length, dx/length)
    
    shrunk_points = []
    n = len(boundary)
    
    for i in range(n):
        p1 = boundary[i]
        p2 = boundary[(i + 1) % n]
        p0 = boundary[(i - 1) % n]
        
        # Get normals for both segments
        n1x, n1y = get_normal(p0, p1)
        n2x, n2y = get_normal(p1, p2)
        
        # Average the normals
        nx = (n1x + n2x) / 2
        ny = (n1y + n2y) / 2
        
        # Normalize the averaged normal
        length = math.sqrt(nx*nx + ny*ny)
        if length > 1e-6:
            nx /= length
            ny /= length
            
            # Move point inward
            shrunk_points.append(Point(
                p1.x + nx * shrink_distance,
                p1.y + ny * shrink_distance
            ))
    
    return shrunk_points

def generate_ironing_pattern(interface: SupportInterface, spacing: float, flow: float, speed: float, settings: dict, shrink_distance: float = 0) -> List[str]:
    if not interface.points:
        print("Skipping interface - no points")
        return []
    
    # Compute boundaries if not already done
    if not interface.boundaries:
        interface.compute_boundaries()
    
    if not interface.boundaries:
        print("Skipping interface - no boundaries detected")
        return []
    
    min_x, max_x, min_y, max_y = interface.bounds
    print(f"\nGenerating ironing pattern:")
    print(f"Bounds: X[{min_x:.2f}, {max_x:.2f}] Y[{min_y:.2f}, {max_y:.2f}]")
    if shrink_distance > 0:
        print(f"Shrinking pattern by {shrink_distance}mm")
    
    gcode = []
    line_width = 0.4  # mm
    line_height = 0.0075  # mm
    
    gcode.append("M204 P1500")
    gcode.append(";TYPE:Ironing")
    gcode.append(f";WIDTH:{line_width}")
    gcode.append(f";HEIGHT:{line_height}")
    gcode.append(f"G1 F{speed}")
    
    # Create shrunk boundaries if needed
    working_boundaries = []
    for boundary in interface.boundaries:
        if shrink_distance > 0:
            shrunk = shrink_boundary(boundary.points, shrink_distance)
            if shrunk:
                working_boundaries.append(Boundary(shrunk, boundary.is_outer))
        else:
            working_boundaries.append(boundary)
    
    if not working_boundaries:
        print("No valid boundaries after shrinking")
        return []
    
    def is_point_inside(x: float, y: float) -> bool:
        # First check grid with circular pattern
        cell_x = int(x / interface.grid_size)
        cell_y = int(y / interface.grid_size)
        radius_cells = int(POINT_SEARCH_RADIUS / interface.grid_size)
        
        # If we're shrinking, only use the boundary check
        if shrink_distance > 0:
            test_point = Point(x, y)
            for boundary in working_boundaries:
                if point_in_polygon(test_point, boundary.points):
                    return True
            return False
        
        # Otherwise use both grid and boundary checks
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx*dx + dy*dy <= radius_cells*radius_cells:
                    if interface.grid.get((cell_x + dx, cell_y + dy), False):
                        return True
        
        test_point = Point(x, y)
        for boundary in working_boundaries:
            if point_in_polygon(test_point, boundary.points):
                return True
        
        return False
    
    # Adjust bounds if we're shrinking
    if shrink_distance > 0:
        min_x += shrink_distance
        max_x -= shrink_distance
        min_y += shrink_distance
        max_y -= shrink_distance
    
    # Generate parallel lines using scanline approach
    y = min_y
    direction = 1
    segments_count = 0
    
    while y <= max_y:
        segments = []
        x = min_x
        inside = False
        last_x = None
        
        while x <= max_x:
            is_inside = is_point_inside(x, y)
            if is_inside != inside:
                if is_inside:
                    last_x = x
                else:
                    if last_x is not None:
                        segments.append((last_x, x))
                inside = is_inside
            x += SCAN_RESOLUTION
        
        if inside and last_x is not None:
            segments.append((last_x, x))
        
        if segments:
            segments_count += len(segments)
            if direction < 0:
                segments.reverse()
            
            first_x, _ = segments[0]
            gcode.append(f"G1 X{first_x:.3f} Y{y:.3f}")
            
            for start_x, end_x in segments:
                length = abs(end_x - start_x)
                e = calculate_extrusion(length, line_height, line_width, flow, settings['filament_diameter'])
                gcode.append(f"G1 X{end_x:.3f} Y{y:.3f} E{e:.5f}")
        
        y += spacing
        direction *= -1
    
    print(f"Generated {segments_count} ironing segments")
    return [line + "\n" for line in gcode] if segments_count > 0 else []

def process_gcode(path: str, flow: float, spacing: float, speed: float, shrink: float) -> int:
    print(f"\nProcessing G-code file: {path}")
    print(f"Parameters: flow={flow}, spacing={spacing}, speed={speed}, shrink={shrink}mm")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"\nError reading file: {e}")
        return 1

    if not any(re.match(r"\s*M83\b", ln) for ln in lines):
        print("File is not in relative extrusion (M83). Aborting.")
        return 1
    
    settings = parse_slicer_settings(lines)
    print(f"\nSlicer settings:")
    print(f"Layer height: {settings['layer_height']}")
    print(f"Extrusion width: {settings['extrusion_width']}")
    print(f"Filament diameter: {settings['filament_diameter']}")
    
    sections = extract_interface_sections(lines)
    
    if not sections:
        print("No support interface sections found.")
        return 0
    
    out = lines.copy()
    offset = 0
    total_sections = 0
    
    for layer_idx in sorted(sections.keys()):
        layer_gcode = []
        print(f"\nProcessing layer at index {layer_idx}")
        for interface in sections[layer_idx]:
            ironing = generate_ironing_pattern(interface, spacing, flow, speed, settings, shrink)
            if ironing:  # Only add if we actually generated some G-code
                layer_gcode.extend(ironing)
                total_sections += 1
        
        if layer_gcode:
            print(f"Adding {len(layer_gcode)} lines of ironing G-code at layer {layer_idx}")
            # Insert just before the layer change
            out[layer_idx + offset:layer_idx + offset] = layer_gcode
            offset += len(layer_gcode)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(out)
    except Exception as e:
        print(f"\nError writing file: {e}")
        return 1

    print(f"\nAdded ironing to {total_sections} support interface sections.")
    return 0

################################################################################
# CLI                                                                          #
################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Iron support interface sections with precise boundary detection (requires relative-E M83).")
    parser.add_argument('input_file')
    parser.add_argument('-flowrate', type=float, default=0.15,
                      help="Flow rate factor for ironing (e.g. 0.15 = 15% of normal flow)")
    parser.add_argument('-spacing', type=float, default=0.10,
                      help="Line spacing for ironing in mm")
    parser.add_argument('-speed', type=float, default=900,
                      help="Speed for ironing in mm/s")
    parser.add_argument('-shrink', type=float, default=0.5,
                      help="Distance to shrink the ironing pattern from boundaries in mm")
    args = parser.parse_args()
    
    result = process_gcode(args.input_file, args.flowrate, args.spacing, args.speed, args.shrink)
    
    # Keep console window open and wait for key press
    print("\nPress any key to exit...")
    if sys.platform == 'win32':
        msvcrt.getch()
    else:
        input()
    
    exit(result)
