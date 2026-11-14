"""moma_pipeline - MotherMachine data analysis pipeline.

ND2 file or FOV TIFF folder -> channel & cell detection -> distance-transform
(DT) morphology -> CSV/pickle exports. Model usage is confined to
`moma_pipeline.models` (delegating to omnishim).
"""

__version__ = "0.1.0"
