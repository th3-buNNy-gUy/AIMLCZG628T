
class KeyGenerator:

    __key__ : int = 0

    @staticmethod
    def generate_key() -> int:
        """
        Generate a unique int key as an identifier for streamlit objects.


        Returns:
            int: A unique key.
        """
        __class__.__key__ += 1
        return __class__.__key__