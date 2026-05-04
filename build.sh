# Build for Apple II
make apple2

## Build to image
cp apple2files/blank.dsk apple2files/program.dsk
/opt/homebrew/opt/openjdk@17/bin/java -jar ac.jar -as apple2files/program.dsk program B 0x0803 < build/program.apple2