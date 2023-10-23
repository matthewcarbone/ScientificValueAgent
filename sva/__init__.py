from dunamai import Version

version = Version.from_any_vcs()
__version__ = version.serialize()
del version


from sva.value import svf  # noqa
