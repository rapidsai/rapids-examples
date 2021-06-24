import codecs # Just a nice way to ensure bytes are always UTF-8, not required just nice to have


def describe(processor):
    processor.setDescription("Generates dummy data")

# Any one time context setup you might want to perform
def onInitialize(processor):
    processor.setSupportsDynamicProperties()

# Reads from a local file and writes the data to a minifi flowfile
class OutputWriter(object):
    def __init__(self, file_path):
        self.file_path = file_path
        f = open(self.file_path, "r")
        self.content = f.read()
        f.close()

    # This function is invoked by the session.write call. It will perform the actual data movement and return the length of bytes moved
    # so the framework can track progress
    def process(self, output_stream):
        codecs.getwriter('utf-8')(output_stream).write(self.content)
        return len(self.content)


# Invoked by the minifi framework, can be setup on a timer or event driven
def onTrigger(context, session):
    # Flowfile travel from processor to processor, since this is the first processor we need to "create" one
    flow_file = session.create()

    # Here we can write literally any data we want into a flowfile. This could be a single message
    # or a batch of them. Its really up to how you want to design the flow
    write_cb = OutputWriter("/minifi/data/tips.csv")

    # These are minifi specific concepts and also NiFi. A Session may also have attributes.
    # You can write data as large as your local disk since this is a outputstream to a local
    # disk, if desired it can also be setup to not have durability and better performance
    # but generally that isn't a good idea unless the data is truly "throw away-able (copyright)"
    session.write(flow_file, write_cb)

    # Here we just naively assume things went ok, in reality you would have other relationship like REL_FAILURE
    # where the flowfile would be routed to that queue/connection if something went wrong
    session.transfer(flow_file, REL_SUCCESS)
