using Gates;

const float eps = 1e-1f;
const float rate = 1e-1f;
Random random = new();
var w1 = (float)random.NextDouble();
var w2 = (float)random.NextDouble();
var b = (float)random.NextDouble();

AndGate andGate = new(
    new float[,] {
        { 0, 0, 0 },
        { 0, 1, 0 },
        { 1, 0, 0 },
        { 1, 1, 1 }
    });
for (var i = 0; i < 100_000; ++i) {
    var c = andGate.Cost(w1, w2, b);
    Console.WriteLine("w1 = {0}, w2 = {1}, b = {2}, cost = {3}", w1, w2, b, c);
    var dw1 = (andGate.Cost(w1 + eps, w2, b) - c) / eps;
    var dw2 = (andGate.Cost(w1, w2 + eps, b) - c) / eps;
    var db = (andGate.Cost(w1, w2, b + eps) - c) / eps;
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
for (var i = 0; i < 100_000; ++i) {
    var c = orGate.Cost(w1, w2, b);
    Console.WriteLine("w1 = {0}, w2 = {1}, b = {2}, cost = {3}", w1, w2, b, c);
    var dw1 = (orGate.Cost(w1 + eps, w2, b) - c) / eps;
    var dw2 = (orGate.Cost(w1, w2 + eps, b) - c) / eps;
    var db = (orGate.Cost(w1, w2, b + eps) - c) / eps;
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
for (var i = 0; i < 1_000; ++i) {
    var c = nandGate.Cost(w1, w2, b);
    Console.WriteLine("w1 = {0}, w2 = {1}, b = {2}, cost = {3}", w1, w2, b, c);
    var dw1 = (nandGate.Cost(w1 + eps, w2, b) - c) / eps;
    var dw2 = (nandGate.Cost(w1, w2 + eps, b) - c) / eps;
    var db = (nandGate.Cost(w1, w2, b + eps) - c) / eps;
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
for (var i = 0; i < 100_000; ++i) {
    var g = xor.Finite_diff(xor, eps);
    xor = XorGate.TrainModel(xor, g, rate);
    Console.WriteLine("cost = {0}", xor.Cost(xor));
}
Console.WriteLine("cost = {0}", xor.Cost(xor));
Console.WriteLine("--------------------------------------------------");
for (var i = 0; i < 2; ++i) {
    for (var j = 0; j < 2; ++j) {
        Console.WriteLine("{0} ^ {1} => {2}", i, j, XorGate.Forward(xor, i, j));
    }
}
Console.WriteLine("--------------------------------------------------");
Console.WriteLine("OR NEURON");
for (var i = 0; i < 2; ++i) {
    for (var j = 0; j < 2; ++j) {
        Console.WriteLine(SigmoidF(xor.OrW1 * i + xor.OrW2 * j + xor.OrB));
    }
}
Console.WriteLine("--------------------------------------------------");
Console.WriteLine("NAND NEURON");
for (var i = 0; i < 2; ++i) {
    for (var j = 0; j < 2; ++j) {
        Console.WriteLine(SigmoidF(xor.NandW1 * i + xor.NandW1 * j + xor.NandB));
    }
}
Console.WriteLine("--------------------------------------------------");
Console.WriteLine("AND NEURON");
for (var i = 0; i < 2; ++i) {
    for (var j = 0; j < 2; ++j) {
        Console.WriteLine(SigmoidF(xor.AndW1 * i + xor.AndW1 * j + xor.AndB));
    }
}

static float SigmoidF(float x) {
    return 1.0f / (1.0f + (float)Math.Exp(-x));
}

