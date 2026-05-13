from AnyQt.QtCore import QSettings

settings = QSettings()
settings.setValue("network/use-certs", True)
