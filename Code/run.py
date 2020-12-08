import network
net = network.Network([1,3,2])
print(zip(net.sizes[:-1],net.sizes[1:]))
for x,y in zip(net.sizes[:-1],net.sizes[1:]):
    print(x)
    print(y)
