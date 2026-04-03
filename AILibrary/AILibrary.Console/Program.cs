while (true)
{
    int input = SystemConsole.ConvertIntoOptionsMode(["AutogradTest", "Credits", "Exit"], "Options/Tests to run: ");

    switch (input)
    {
        case 0:
            AutogradTest.Test();
            break;

        case 1:
            SystemConsole.DisplayCredits();
            break;

        default:
            Environment.Exit(0);
            break;
    }
}
