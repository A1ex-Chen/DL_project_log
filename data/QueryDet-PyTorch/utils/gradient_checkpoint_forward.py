def forward(input):
    for j in range(start, end + 1):
        input = functions[j](input)
    return input
