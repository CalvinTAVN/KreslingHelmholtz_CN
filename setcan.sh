
echo Setting up CanBus
sudo modprobe can
sudo modprobe mttcan
sudo modprobe can_raw
sudo ip link set can1 up type can bitrate 1000000
echo Finished Setting up CanBus
