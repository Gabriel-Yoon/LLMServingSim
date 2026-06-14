import re
from .logger import get_logger

class Controller():
    def __init__(self, total_num):
        self.end_dict = {}
        self.total_num = total_num
        self.logger = get_logger(self.__class__)
        for i in range(total_num):
            self.end_dict[i] = -1


    def read_wait(self, p):
        out = [""]
        while "Waiting" not in out[-1] and out[-1] != "Checking Non-Exited Systems ...\n":
            line = p.stdout.readline()
            if line == "":
                break
            out.append(line)
        return out

    def check_end(self, p):
        # After the main loop sends a single "exit", ASTRA-Sim breaks out of its
        # while(!exit) loop and prints a terminal handshake line. But in the
        # degenerate EP case where only one instance ever ran real work (e.g.
        # N=1 over a 32-way EP group, with the other ranks running dummy waves),
        # ASTRA's exit can land while one or more NPUs are still blocked on
        # std::getline(std::cin) — they keep re-emitting a "Waiting" prompt that
        # nobody answers. A read-only check_end then deadlocks; if instead every
        # remaining NPU is asleep, ASTRA spins printing check lines and we grow
        # ``out`` until the OS OOM-kills us (SIGKILL/137).
        #
        # Fix: keep driving ASTRA toward termination. Whenever it prompts for
        # input again, answer "exit" so the next getline breaks its loop; stop
        # on either terminal line or on EOF. ``out`` is kept bounded so a runaway
        # ASTRA can never OOM us. In the normal path (N>=2) ASTRA breaks
        # immediately, no "Waiting" prompt appears, and this behaves as before.
        out = ["", ""]
        while out[-2] != "All Request Has Been Exited\n" and out[-2] != "ERROR: Some Requests Remain\n":
            line = p.stdout.readline()
            # EOF: ASTRA-Sim closed stdout (process exited) before emitting a
            # terminal handshake line. Mirror read_wait()'s EOF handling.
            if line == "":
                self.logger.warning(
                    "ASTRA-Sim stdout closed during exit handshake before "
                    "'All Request Has Been Exited' — treating simulation as "
                    "ended (Python-side request accounting already complete)."
                )
                break
            # ASTRA is still blocked waiting for a per-NPU command. Push it to
            # exit instead of letting it (and us) hang.
            if "Waiting" in line:
                self.write_flush(p, "exit")
            out.append(line)
            # Keep only a small tail; we only ever inspect out[-2]/out[-4].
            if len(out) > 64:
                out = out[-64:]
            p.stdout.flush()
        if len(out) >= 4:
            print(out[-4], end='')
        print(out[-2], end='')
        return out

    def write_flush(self, p, input):
        # For debugging
        # print(input)
        p.stdin.write(input+'\n')
        p.stdin.flush()
        return

    def parse_output(self, output):
        pattern = r"sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles, exposed communication (\d+) cycles."
        match = re.search(pattern, output)
        if match:
            sys = int(match.group(1))
            id = int(match.group(2))
            cycle = int(match.group(3))
            com_cycle = int(match.group(4))

            if self.end_dict[sys] != id:
                self.logger.info(
                    "NPU[%d] iteration %d finished, %d cycles, exposed communication %d cycles.",
                    sys,
                    id,
                    cycle,
                    com_cycle,
                )
                self.end_dict[sys] = id
            return {'sys': sys, 'id': id, 'cycle': cycle}
        return