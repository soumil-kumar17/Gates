using Gates;
const float EPS = 1e-1f;
const float rate = 1e-1f;
Random random = new();
float w1 = (float)random.NextDouble();
float w2 = (float)random.NextDouble();
float b = (float)random.NextDouble();

AndGate andGate = new(
    new float[,] {
        { 0, 0, 0 },
        { 0, 1, 0 },
        { 1, 0, 0 },
        { 1, 1, 1 }
    });
for (int i = 0; i < 5_000; ++i) {
    float c = andGate.Cost(w1, w2, b);
    Console.WriteLine("w1 = {0}, w2 = {1}, b = {2}, cost = {3}", w1, w2, b, c);
    float dw1 = (andGate.Cost(w1 + EPS, w2, b) - c) / EPS;
    float dw2 = (andGate.Cost(w1, w2 + EPS, b) - c) / EPS;
    float db = (andGate.Cost(w1, w2, b + EPS) - c) / EPS;
    w1 -= rate * dw1;
    w2 -= rate * dw2;
    b -= rate * db;
}

OrGate orGate = new(
    new float[,] {
        { 0, 0, 0 },
        { 0, 1, 1 },
        { 1, 0, 1 },
        { 1, 1, 1 }
    });
for (int i = 0; i < 100_000; ++i) {
    float c = orGate.Cost(w1, w2, b);
    Console.WriteLine("w1 = {0}, w2 = {1}, b = {2}, cost = {3}", w1, w2, b, c);
    float dw1 = (orGate.Cost(w1 + EPS, w2, b) - c) / EPS;
    float dw2 = (orGate.Cost(w1, w2 + EPS, b) - c) / EPS;
    float db = (orGate.Cost(w1, w2, b + EPS) - c) / EPS;
    w1 -= rate * dw1;
    w2 -= rate * dw2;
    b -= rate * db;
}

NandGate nandGate = new(
    new float[,] {
        { 0, 0, 1 },
        { 0, 1, 1 },
        { 1, 0, 1 },
        { 1, 1, 0 }
    });
for (int i = 0; i < 1_000; ++i) {
    float c = nandGate.Cost(w1, w2, b);
    Console.WriteLine("w1 = {0}, w2 = {1}, b = {2}, cost = {3}", w1, w2, b, c);
    float dw1 = (nandGate.Cost(w1 + EPS, w2, b) - c) / EPS;
    float dw2 = (nandGate.Cost(w1, w2 + EPS, b) - c) / EPS;
    float db = (nandGate.Cost(w1, w2, b + EPS) - c) / EPS;
    w1 -= rate * dw1;
    w2 -= rate * dw2;
    b -= rate * db;
}

//Xor cost fn does not go below 0.25 using only 1 neuron.
//Can act as any logic gate, provided correct input is given.
XorGate xor = new(
    new float[,] {
        {0, 0, 0 },
        {0, 1, 1 },
        {1, 0, 1 },
        {1, 1, 0 }
    });

XorGate.Init_Rand_Xor(xor);
for (int i = 0; i < 100_000; ++i) {
    XorGate g = xor.Finite_diff(xor, EPS);
    xor = XorGate.TrainModel(xor, g, rate);
    Console.WriteLine("cost = {0}", xor.Cost(xor));
}
Console.WriteLine("cost = {0}", xor.Cost(xor));
Console.WriteLine("--------------------------------------------------");
for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
        Console.WriteLine("{0} ^ {1} => {2}", i, j, XorGate.Forward(xor, i, j));
    }
}
Console.WriteLine("--------------------------------------------------");
Console.WriteLine("OR NEURON");
for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
        Console.WriteLine(Sigmoid_f(xor.Or_w1 * i + xor.Or_w2 * j + xor.Or_b));
    }
}
Console.WriteLine("--------------------------------------------------");
Console.WriteLine("NAND NEURON");
for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
        Console.WriteLine(Sigmoid_f(xor.Nand_w1 * i + xor.Nand_w1 * j + xor.Nand_b));
    }
}
Console.WriteLine("--------------------------------------------------");
Console.WriteLine("AND NEURON");
for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
        Console.WriteLine(Sigmoid_f(xor.And_w1 * i + xor.And_w1 * j + xor.And_b));
    }
}

static float Sigmoid_f(float x) {
    return 1.0f / (1.0f + (float)Math.Exp(-x));
}

