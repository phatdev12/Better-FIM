def fitness_F(MF, DCV, lamda=0.5):
  return lamda * MF - (1 - lamda) * DCV
