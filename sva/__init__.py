# DO NOT CHANGE BELOW ---------------------------------------------------------
# ALSO DO NOT TYPE dunder version ANYWHERE ABOVE THIS
# This is replaced at build time automatically during deployment and
# installation. Replacing anything will mess that up and crash the entire
# build.
__version__ = "dev"  # semantic-version-placeholder
# DO NOT CHANGE ABOVE ---------------------------------------------------------

# Silly hack. Useful for local development
if __version__ == "dev":
    try:
        from dunamai import Version

        version = Version.from_any_vcs()
        __version__ = version.serialize()
        del version
    except ImportError:
        print("You are running a local copy of SVA (not installed via pip)")
        print(
            "__version__ = 'dev'; pip install dunamai to track local version"
        )
        pass
