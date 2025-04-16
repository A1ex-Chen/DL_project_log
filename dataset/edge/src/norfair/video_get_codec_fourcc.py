def get_codec_fourcc(self, filename: str) ->Optional[str]:
    if self.output_fourcc is not None:
        return self.output_fourcc
    extension = filename[-3:].lower()
    if 'avi' == extension:
        return 'XVID'
    elif 'mp4' == extension:
        return 'mp4v'
    else:
        self._fail(
            f"""[bold red]Could not determine video codec for the provided output filename[/bold red]: [yellow]{filename}[/yellow]
Please use '.mp4', '.avi', or provide a custom OpenCV fourcc codec name."""
            )
        return None
