from setuptools import setup, Extension
from Cython.Build import cythonize

# Define shared macros
#shared_macros = [('CYTHON_TRACE', '1')]
shared_macros = []

# Define your extensions with shared macros
extensions = [
    Extension("string_builder", ["string_builder.pyx"], define_macros=shared_macros),
    Extension("utils", ["utils.pyx"], define_macros=shared_macros),
    Extension("globalvars", ["globalvars.pyx"], define_macros=shared_macros),
    Extension("pool", ["pool.pyx"], define_macros=shared_macros),
    Extension("picture", ["picture.pyx"], define_macros=shared_macros),
    Extension("io_utils", ["io_utils.pyx"], define_macros=shared_macros),
    Extension("draw_utils", ["draw_utils.pyx"], define_macros=shared_macros),
    Extension("solver", ["solver.pyx"], define_macros=shared_macros),
    Extension("tests", ["tests.pyx"], define_macros=shared_macros)
]

# Define compiler directives
compiler_directives = {
    'profile': False,
    'language_level': 3,
    'boundscheck': False,
    'cdivision': True,
    'initializedcheck': False,
    'infer_types': True,
    'nonecheck': False,
    'linetrace': False,
    'binding': False,
    'wraparound': False,
    'annotation_typing': True
}

setup(
    name="YourProjectName",
    ext_modules=cythonize(extensions, compiler_directives=compiler_directives)
)
