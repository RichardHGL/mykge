from util.parser import args_parser
from agent import BaseAgent, KbganAgent


def main():
    args = args_parser()
    if args.model == 'Kbgan':
        agent = KbganAgent(args)
    else:
        agent = BaseAgent(args)
    if args.eval:
        agent.evaluate_once()
    else:
        agent.train()


if __name__ == '__main__':
    main()
