from math import cos

def find_coord(n, m , center_lat, center_long, row, col):
    lat_step = 1 / 296
    long_step = 375 / (111000 * cos(lat_step))
    zero_lat, zero_long = center_lat - (n // 2) * lat_step - lat_step / 2 * (n % 2), center_long - (m // 2) * long_step - long_step / 2 * (m % 2)
    lat = zero_lat + row * lat_step
    long = zero_long + col * long_step
    return lat, long

def build_grid(n, m, center_lat, center_long):
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            lat, long = find_coord(n, m, center_lat, center_long, i, j)
            row.append((lat, long))
        grid.append(row)
    return grid

if __name__ == "__main__":
    n, m = 5, 5
    center_lat, center_long = 43.4723, -80.5449
    grid = build_grid(n, m, center_lat, center_long)
    for row in grid:
        print(row)