
class Colab:
    @staticmethod
    def ClearOutput():
        from IPython.display import clear_output
        clear_output()


    @staticmethod
    def Terminate():
        from google.colab import runtime
        runtime.unassign()
