def find_coord(n, m , center_lat, center_long, row, col):
    lat_step = 0.01
    long_step = 0.01
    zero_lat, zero_long = center_lat - (n // 2) * lat_step - lat_step / 2 * (n % 2), center_long - (m // 2) * long_step - long_step / 2 * (m % 2)
    lat = zero_lat + row * lat_step
    long = zero_long + col * long_step

    return lat, long