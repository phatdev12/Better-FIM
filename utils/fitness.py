def fitness_F(MF, DCV, lamda=0.5):
    """
    Hàm fitness kết hợp MF và DCV
    
    Args:
        MF: Min Fraction - coverage tối thiểu (càng cao càng tốt)
        DCV: Deviation from Coverage Violation - độ lệch (càng thấp càng tốt)
        lamda: trọng số cân bằng giữa MF và DCV (0-1)
    
    Returns:
        fitness score (càng cao càng tốt)
    """
    return lamda * MF - (1 - lamda) * DCV
