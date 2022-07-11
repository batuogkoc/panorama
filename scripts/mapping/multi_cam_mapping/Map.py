from abc import ABC, abstractmethod

class Map(ABC):
    @abstractmethod
    def get_image():
        pass

    @abstractmethod
    def global_meters_to_local_pixels(global_meter_coordinates):
        pass
    
    @abstractmethod
    def local_pixels_to_global_meters(local_pixel_values):
        pass
