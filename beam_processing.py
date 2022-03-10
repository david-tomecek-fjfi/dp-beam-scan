from typing import List, Union
from optix.beams import GaussianBeam
import numpy as np
TOLERANCE = 1


#TODO: firnin upozorneni, kdyz je M parameter nad nakou mez
def extract(positions: List[float], data: List[np.ndarray], wave_length, pixel_size, **kwargs) -> GaussianBeam:
    """From image data and its position extract the gaussian beam

    Args:
        positions (List[float]): positions of taken image data (arbitrary coordinate system)
        data (List[np.ndarray]): image data matching (sorted in the same fassion as positions array is)
        wave_length (_type_): wave_length of the gaussian beam
        pixel_size (_type_): pixel size of the camera that took the images (for unit preserving)

    Returns:
        GaussianBeam
    """
    amplitudes = []
    radiuses = []
    for dataInput in data:
        ampl, rad = _extract_amplitude_radius(dataInput)
        amplitudes.append(ampl)
        radiuses.append(rad * pixel_size)
    divergence, w_loc = _calculate_div_waist_locations(positions, radiuses)
    return GaussianBeam(
        wave_length=wave_length, 
        amplitude=kwargs.get("amplitude", 1), 
        refractive_index=kwargs.get("n",1), 
        waist_location=w_loc,
        divergence=divergence)

def _extract_amplitude_radius(data: np.ndarray) -> Union[float, float]:
    amplitude = np.amax(data)
    THRESHOLD = int(amplitude / np.e**2)
    Y_MAX, X_MAX = np.unravel_index(data.argmax(), data.shape)
    is_border_point = lambda a: abs(a - THRESHOLD) <= TOLERANCE
    get_radius = lambda x, y: np.sqrt((X_MAX - x)**2 + (Y_MAX - y)**2)
    radiuses = []
    for y in range(len(data)):
        for x in range(len(data[0])):
            if is_border_point(data[y][x]):
                radiuses.append(get_radius(x, y))
    return amplitude, np.average(radiuses)     

def _calculate_div_waist_locations(positions, beam_radiuses) -> Union[float, float]:
    a, b = np.polyfit(positions, beam_radiuses, deg=1)
    div = np.arctan(a)
    w_loc = -b/a
    return div, w_loc


if __name__ == "__main__":
    pass

