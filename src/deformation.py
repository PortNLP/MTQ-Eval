# src/deformation.py
import random

def swap_words(text: str, num_swaps: int = 4) -> str:
    """Randomly swaps words in the text to degrade coherence."""
    words = text.split()
    length = len(words)
    if length < 2:
        return text
    
    # Ensure we don't swap more than possible
    actual_swaps = min(num_swaps, length // 2)
    
    for _ in range(actual_swaps):
        idx1, idx2 = random.sample(range(length), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        
    return ' '.join(words)

def deform_text(text: str, method: str = 'swap') -> str:
    if method == 'swap':
        # Randomly swap between 3 to 6 words as per paper
        return swap_words(text, random.randint(3, 6))
    return text
