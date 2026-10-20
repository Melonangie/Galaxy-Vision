# Place your own data /data

# Build the image
docker build --platform linux/386 -t starlight:v04 .

# Run the container
docker run --rm -it --platform linux/386 \
  -v "$PWD/data:/data" \
  starlight:v04

# Run the built-in example
cd STARLIGHTv04/
./StarlightChains_v04.exe < grid_example1.in

# Run it in one command (batch mode)
docker run --rm --platform linux/386 -v "$PWD/data:/data" -w /data \
  starlight:v04 sh -c "starlight < my_grid.in"
