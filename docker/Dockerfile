# 32-bit userland to match the i386 static binary
FROM --platform=linux/386 debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends bzip2 tar \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /starlight

# Unpack the distribution (bases, masks, example spectra, config, grid files)
COPY StarlightChains_v04.Distrib.tar.bz2 /tmp/
RUN tar -xjf /tmp/StarlightChains_v04.Distrib.tar.bz2 -C /starlight \
    && rm /tmp/StarlightChains_v04.Distrib.tar.bz2

# The executable
COPY StarlightChains_v04.i386_g77-3.3.5_static.exe /starlight/STARLIGHTv04/StarlightChains_v04.exe
RUN chmod +x /starlight/STARLIGHTv04/StarlightChains_v04.exe

RUN sed -i "s|/home/cid/STARLIGHTv04|/starlight/STARLIGHTv04|g" /starlight/STARLIGHTv04/grid_example1.in

# Your own data/results can be mounted here
VOLUME ["/data"]

#CMD ["bash", "-c", "cd STARLIGHTv04 &&  ./StarlightChains_v04.exe < grid_example1.in"]
CMD ["bash"]
