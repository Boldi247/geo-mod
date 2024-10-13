from bezier.bezier import bezier_main

def main():
    print("To diplay bezier surface, type in '1'")
    choice = input()

    if choice == '1':
        print("Displaying Bezier Surface")
        bezier_main()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()